shogunGP <- function(
	x, y, m = NULL,
	kernel ='rbfdot', kpar = list(sigma = 'automatic'), 
	var = 1
	) {

	require(shogun);
	requireNamespace(kernlab);
	sigma <- ifelse('automatic' == kpar$sigma, kernlab::sigest(x)[[2]], kpar$sigest);

	x <- scale(x);
	y <- scale(y);
	feats <- RealFeatures();
	dump <- feats$set_feature_matrix(t(x));
	labels <- RegressionLabels();
	dump <- labels$set_labels(y);

	if (!is.null(m)) {
		feats_inducing <- RealFeatures();
		dump <- feats_inducing$set_feature_matrix(t(x[sample(nrow(x), size = m), ]));

		inf <- FITCInferenceMethod(GaussianKernel(10, 1/sigma), feats, ZeroMean(), labels, GaussianLikelihood(var), feats_inducing);
	} else {
		inf <- ExactInferenceMethod(GaussianKernel(10, 1/sigma), feats, ZeroMean(), labels, GaussianLikelihood(var));
		}
			
	start <- Sys.time();
	gp <- GaussianProcessRegression(inf);
	gp$train();
	print(Sys.time() - start);

	return(
		structure(
			list(fit = gp, standardize = list(x = attributes(x)[c('scaled:center', 'scaled:scale')], y = attributes(y)[c('scaled:center', 'scaled:scale')])),
			class = 'shogunGP'
			)
		);
	}

predict.shogunGP <- function(object, newdata) {
	require(shogun);
	newdata <- scale(newdata, center = object$standardize$x[['scaled:center']], scale = object$standardize$x[['scaled:scale']]);
	feats_test <- RealFeatures();
	dump <- feats_test$set_feature_matrix(t(newdata));
	out <- object$fit$apply(feats_test)$get_labels();
	return(out*object$standardize$y[['scaled:scale']] + object$standardize$y[['scaled:center']]);
	}



dat <- read.csv('Documents/olfaction_intensity.csv');
dat <- as.matrix(dat);

fit <- shogunGP(dat[, -1], dat[, 1], m = 10);
pred <- predict(fit, newdata = dat[, -1]);
