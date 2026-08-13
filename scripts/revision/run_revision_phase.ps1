param(
  [Parameter(Mandatory = $true)][string]$GridPath,
  [Parameter(Mandatory = $true)][ValidateSet("bart_transformed", "boosting", "kernel", "cdc", "coverage")][string]$Phase,
  [Parameter(Mandatory = $true)][string]$RLibrary,
  [Parameter(Mandatory = $true)][string]$OutputRoot,
  [Parameter(Mandatory = $true)][ValidateSet("YES")][string]$ConfirmCanonical,
  [int]$MaxWorkers = 4,
  [int]$StatusIntervalMinutes = 15,
  [string]$Rscript = "Rscript.exe"
)

$ErrorActionPreference = "Stop"
if ($MaxWorkers -lt 1 -or $MaxWorkers -gt 12) { throw "MaxWorkers must be 1 through 12" }
if ($Phase -eq "bart_transformed" -and $MaxWorkers -gt 8) {
  throw "The transformed BART runner permits at most 8 workers"
}
$grid = Resolve-Path -LiteralPath $GridPath
$expectedHashPath = [System.IO.Path]::ChangeExtension($grid.Path, ".sha256")
if (-not (Test-Path -LiteralPath $expectedHashPath)) { throw "Missing run-plan SHA-256 file" }
$expectedHash = (Get-Content -LiteralPath $expectedHashPath -Raw).Trim().ToLowerInvariant()
$actualHash = (Get-FileHash -LiteralPath $grid.Path -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw "Run-plan SHA-256 mismatch" }

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$library = Resolve-Path -LiteralPath $RLibrary
$root = [System.IO.Path]::GetFullPath($OutputRoot)
[System.IO.Directory]::CreateDirectory($root) | Out-Null
$logRoot = Join-Path $root ("launcher-logs\" + $Phase)
[System.IO.Directory]::CreateDirectory($logRoot) | Out-Null
$rows = @(Import-Csv -LiteralPath $grid.Path | Where-Object { $_.phase -eq $Phase })
if ($rows.Count -eq 0) { throw "No jobs for phase: $Phase" }

function Get-JobArguments($row) {
  $script = Join-Path $repoRoot.Path $row.script
  $common = @($script, "--mode=canonical", "--confirm-canonical=YES", "--resume=true")
  if ($Phase -eq "bart_transformed") {
    $commit = (& git -C $repoRoot.Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch "^[0-9a-f]{40}$") {
      throw "Unable to determine the full Git commit"
    }
    return @($script, "--mode=canonical", "--confirm-canonical=YES",
      "--output-dir=$(Join-Path $root 'bart-transformed')", "--code-commit=$commit",
      "--workers=$MaxWorkers")
  }
  if ($Phase -eq "coverage") {
    $args = $common + @("--seed=$($row.seed)", "--n0=$($row.n0)", "--n1=$($row.n1)",
      "--batts-lib=$($library.Path)", "--output-dir=$(Join-Path $root 'coverage')")
    if ($row.scenario) { $args += "--scenario=$($row.scenario)" }
    if ($row.family -eq "20d") { $args += "--transformed=$($row.transformed.ToLowerInvariant())" }
    return $args
  }
  if ($Phase -eq "boosting") {
    return $common + @("--family=20d", "--scenario=$($row.scenario)", "--seed=$($row.seed)",
      "--n0=$($row.n0)", "--n1=$($row.n1)", "--transformed=$($row.transformed.ToLowerInvariant())",
      "--r-lib=$($library.Path)", "--output-dir=$(Join-Path $root 'comparators\boosting')")
  }
  if ($Phase -eq "kernel") {
    return $common + @("--method=$($row.method)", "--scenario=$($row.scenario)", "--seed=$($row.seed)",
      "--n0=$($row.n0)", "--n1=$($row.n1)", "--transformed=$($row.transformed.ToLowerInvariant())",
      "--r-lib=$($library.Path)", "--output-dir=$(Join-Path $root 'comparators\kernel')")
  }
  $input = Join-Path $root ("comparators\boosting\canonical\" + $row.dependency_id + ".rds")
  if (-not (Test-Path -LiteralPath $input)) { throw "Missing CDC dependency: $input" }
  return @($script, "--resume=true", "--boosting-result=$input",
    "--output-dir=$(Join-Path $root 'comparators\cdc')")
}

$queue = [System.Collections.Generic.Queue[object]]::new()
foreach ($row in $rows) { $queue.Enqueue($row) }
$running = @{}
$completed = 0
$failed = 0
$started = Get-Date
$nextStatus = $started

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  while ($queue.Count -gt 0 -and $running.Count -lt $MaxWorkers) {
    $row = $queue.Dequeue()
    $stdout = Join-Path $logRoot ($row.job_id + ".stdout.log")
    $stderr = Join-Path $logRoot ($row.job_id + ".stderr.log")
    $process = Start-Process -FilePath $Rscript -ArgumentList (Get-JobArguments $row) `
      -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $running[$process.Id] = [pscustomobject]@{ Process = $process; Row = $row; Start = Get-Date }
  }
  Start-Sleep -Seconds 5
  foreach ($id in @($running.Keys)) {
    $entry = $running[$id]
    if ($entry.Process.HasExited) {
      if ($entry.Process.ExitCode -eq 0) { $completed++ } else { $failed++ }
      $running.Remove($id)
    }
  }
  $now = Get-Date
  if ($now -ge $nextStatus) {
    $elapsedHours = [Math]::Max(($now - $started).TotalHours, 1.0 / 3600.0)
    if ($Phase -eq "bart_transformed") {
      $unitTotal = 200
      $unitCompleted = @(Get-ChildItem -LiteralPath (Join-Path $root 'bart-transformed') `
        -Filter '*.done.rds' -File -ErrorAction SilentlyContinue).Count
    } else {
      $unitTotal = $rows.Count
      $unitCompleted = $completed
    }
    $rate = $unitCompleted / $elapsedHours
    $remaining = $unitTotal - $unitCompleted - $failed
    $etaHours = if ($rate -gt 0) { $remaining / $rate } else { [double]::PositiveInfinity }
    $line = "{0:o} phase={1} completed={2}/{3} failed={4} active={5} remaining_hours={6}" -f `
      $now, $Phase, $unitCompleted, $unitTotal, $failed, $running.Count,
      $(if ([double]::IsInfinity($etaHours)) { "estimating" } else { "{0:N2}" -f $etaHours })
    $line | Tee-Object -FilePath (Join-Path $logRoot "phase-status.log") -Append
    $nextStatus = $now.AddMinutes($StatusIntervalMinutes)
  }
}
if ($failed -gt 0) { throw "$failed jobs failed; inspect $logRoot" }
