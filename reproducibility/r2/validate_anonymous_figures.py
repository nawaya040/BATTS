"""Regenerate the documented technical figures in a new external output tree."""
from pathlib import Path
import argparse,subprocess,os,json,time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rscript',required=True);p.add_argument('--drawing-library',type=Path,required=True)
    a=p.parse_args();root=a.candidate.resolve();out=a.out.resolve()
    if out.exists() or out.is_relative_to(root):raise SystemExit('Choose a new output tree outside the release')
    out.mkdir(parents=True)
    env=os.environ.copy();env.update(LANG='C',LC_ALL='C')
    env['R_LIBS_USER']=str(a.drawing_library.resolve())+';'+str(Path(os.environ['LOCALAPPDATA'])/'R/win-library/4.5')
    records=[]
    def run(name,args):
        start=time.monotonic()
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run([a.rscript,'--vanilla',*map(str,args)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(step=name,returncode=r.returncode,elapsed_seconds=time.monotonic()-start))
        (out/'execution.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(name, r.returncode, flush=True)
        if r.returncode:raise SystemExit('Failed '+name+'; see its log')
    fig='code/r2/scripts/figures/'
    for name,preset in [('figure1_main','balanced-sample-size'),('figure1_supp','unbalanced-sample-size')]:
        run(name,[fig+'plot_figure1_r2_preview.R','--results-root=reference/r1/output/section34_1d/full',f'--png-output={out/name/(name+".png")}',f'--pdf-output={out/name/(name+".pdf")}',f'--setting-preset={preset}'])
    run('figure2',[fig+'plot_figure2_r2_preview.R',f'--png-output={out/"figure2/figure2.png"}',f'--pdf-output={out/"figure2/figure2.pdf"}'])
    detail='results/r2/figure_details/'
    run('figure3_source',[fig+'plot_figure3_compact_preview.R','--legacy-detail='+detail+'2d/local_shift/local_shift_5000_5000_5_1_details.rds','--boosting-result='+detail+'boosting/boosting_selection_2d_local_shift_n0-5000_n1-5000_transformed-false_seed-001.rds','--boosting-checksums='+detail+'boosting/output_checksums.csv',f'--output-dir={out/"figure3_source"}'])
    summary='results/r2/summaries/'
    run('figure4',[fig+'plot_figure4_compact_preview.R',summary+'raw-global-null-20260813T162558',summary+'coverage-summary-20260903T170144JST',out/'figure4'])
    run('figure5',['code/portable/plot_figure5.R','results/r2/figure5/figure_20d_localization_r2_preview_summary.csv',out/'figure5'])
    run('supplement_s1',['code/portable/plot_supplement_s1.R','--input-root=results/r2/supplement_s1/boosting',f'--output-dir={out/"supplement_s1"}'])
    run('surfaces_source',[fig+'plot_supplement_eight_panel_release.R',detail+'2d',detail+'20d/latent_location_shift_5000_5000_5_21_details.rds',detail+'boosting',detail+'boosting/output_checksums.csv',out/'surfaces_source'])
    run('surfaces_final',[fig+'plot_eight_panel_equal_size_release.R',out/'figure3_source/plotted_data.rds',out/'surfaces_source',out/'surfaces_final'])
    run('supplement_s4',['code/portable/plot_supplement_s4.R',summary+'coverage-summary-20260903T170144JST',out/'supplement_s4'])
    run('supplement_s6',[fig+'plot_supplement_s6_compact_preview.R',out/'supplement_s6'])
if __name__=='__main__':main()
