
# make moons
set.seed(10)

n = 1000
sd_moon = 0.15

theta_vec = runif(n, 0, pi)

data = matrix(NA, nrow = n, ncol = 2)
a = 0.3

for(i in 1:n){
  x = c(cos(theta_vec[i]), sin(theta_vec[i])) * rnorm(1,mean=1, sd = sd_moon)

  if(runif(1) < 0.5){
   y = x
  }else{
    y = c(1-x[1], -x[2] + a)
  }

  data[i,] = y
}

plot(data[,1], data[,2])

# density?

x.grid = seq(-1.5,2.5,length.out = 100)
y.grid = seq(-1.5+a,1.5,length.out = 100)
grid.points = as.matrix(expand.grid(x.grid,y.grid))

densities = numeric(nrow(grid.points))

for(i in 1:nrow(grid.points)){
  # 1st moon
  x1 = grid.points[i,1]
  x2 = grid.points[i,2]
  r = sqrt(x1^2 + x2^2)
  #theta = acos(x1 / r)
  dens1 = (grid.points[i,2] > 0) * 1 / pi  * dnorm(r, 1, sd_moon)

  # 2nd moon
  x1 = 1 - grid.points[i,1]
  x2 = - grid.points[i,2] + a
  r = sqrt(x1^2 + x2^2)
  #theta = acos(x1 / r)
  dens2 = (grid.points[i,2] < a) * 1 / pi  * dnorm(r, 1, sd_moon)

  densities[i] = 1/2 * dens1 + 1/2 * dens2
}

plot(densities)

densities.df = data.frame(x1 = grid.points[,1], x2 = grid.points[,2], dens = densities)
#densities.df = densities.df[sample(1:nrow(densities.df)),]

ggplot(densities.df, aes(x = x1, y = x2, color = dens)) +
  geom_point(size=3, alpha=0.75)

