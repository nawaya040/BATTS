library(densityratio)

head(numerator_data)
head(denominator_data)

fit = spectral(
  denominator_data$x5,
  numerator_data$x5,
)


class(fit)
summary(fit)
plot(fit)

predict(fit)
