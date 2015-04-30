#' @title Fit Gaussian process regression 
#' @description 
#' Fit Gaussian process regression using shogun's \code{FITCInferenceMethod} and \code{ExactInferenceMethod}
#' @param formula: An R formula
#' @param data: An data frame
#' @param x: A design matrix
#' @param y: A response vector
#' @param m: Dimension of the low rank approach to the covariance matrix: V_nxn = W_nxm W'_mxn + delta*I_nxn
#' @param kernel: Only RBF kernel available
#' @param kpar: A list of kernel parameters
#' @param var: Gaussian noise for regression
shogunGP <- function(x, ...) UseMethod('shogunGP');

shogunGP.formula <- function(formula, data, ...) {
	mf <- model.frame(formula, data);
	x <- model.matrix(formula, data);
	y <- model.response(mf);
	xint <- match("(Intercept)", colnames(x), nomatch = 0);
	if (xint > 0) x <- x[, -xint, drop = FALSE];
	rm(mf);
	out <- shogunGP.default(x, y, ...);
	out$formula <- formula;
	class(out) <- c('shogounGP.formula', class(out));
	return(out);
	}

shogunGP.default <- function(
	x, y, m = NULL,
	kernel ='rbfdot', kpar = list(sigma = 'automatic'), 
	var = 1
	) {

	require(shogun);
	requireNamespace('kernlab');
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

predict.shogunGP.formula <- function(object, newdata, ...) {
	newdata <- model.matrix(object$formula[-2], newdata);
	xint <- match("(Intercept)", colnames(newdata), nomatch = 0);
	if (xint > 0) newdata <- newdata[, -xint, drop = FALSE];
	return(predict.shogunGP(object, newdata));
	}

predict.shogunGP <- function(object, newdata, ...) {
	require(shogun);
	newdata <- scale(newdata, center = object$standardize$x[['scaled:center']], scale = object$standardize$x[['scaled:scale']]);
	feats_test <- RealFeatures();
	dump <- feats_test$set_feature_matrix(t(newdata));
	out <- object$fit$apply(feats_test)$get_labels();
	return(out*object$standardize$y[['scaled:scale']] + object$standardize$y[['scaled:center']]);
	}
