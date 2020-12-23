%%%%%%%%%%%%%%%%%%%%
% Learning systems %
% vt 2019          %
%%%%%%%%%%%%%%%%%%%%

clear all
close all


%% To import the data sets
load('cnDieselTrain.mat')

% To form the training matrix
TotalTrain = cnTrainX';
TotalTrain(:,402) = cnTrainY';

% To look at the data sets
Min_value_cnTestX = min(min(cnTestX));
Max_value_cnTestX = max(max(cnTestX));
Min_value_cnTrainY = min(cnTrainY);
Max_value_cnTrainY = max(cnTrainY);
Mean_value_cnTrainY = mean(cnTrainY);

% To plot the cetane numbers of the training set
figure;
plot(cnTrainY, 'r*')
title('The training set')
xlabel('Sample number')
ylabel('Cetane number')
xlim([0 134])


%% The PCA part
% To train the model from regression learner app
[trainedModel, validationMSE, validationRMSE] = regressionLearnerAppModel(TotalTrain);

% To print the validation results
%Validation_MSE = validationMSE
%Validation_RMSE = validationRMSE

% To predict the cetane numbers for cnTestX using regression learner app
cnTestYPCA1 = trainedModel.predictFcn(cnTestX');

% Manual model
% PCA
[coeff,score,latent,tsquared,explained] = pca(cnTrainX');
First_ten_principal_components = explained(1:10)';
cnTrainXPCA = cnTrainX'*coeff;

% To plot the first principal component and the cetane number
figure;
plot(score(:,1), cnTrainY', 'b*')
title('PCA')
xlabel('First principal component')
ylabel('Cetane number')

% To plot validation and training errors of PCA model
for mm = 1:30
    % To perform cross-validation, and return average MSE across folds
    fcn = @(Xtr, Ytr, Xte) predict(fitlm(Xtr,Ytr), Xte);
    PCAValidationMSE(mm) = crossval('mse', cnTrainXPCA(:,1:mm), cnTrainY','Predfun',fcn, 'kfold',5);
    % To get training errors
    mdl = fitlm(cnTrainXPCA(:,1:mm),cnTrainY');
    cnTrainPCATest = predict(mdl,cnTrainXPCA(:,1:mm));
    PCATrainMSE(mm) = (sum((cnTrainY'-cnTrainPCATest).^2))/133; 
end
figure;
plot(1:mm,PCAValidationMSE, 'ro-');
hold on;
plot(1:mm,PCATrainMSE, 'bo-');
title('Linear regression model using PCA')
legend('Validation MSE','Training MSE')
xlabel('Number of PCA-components')
ylim([0 15])
hold off;

% To train the manual model
mdl = fitlm(cnTrainXPCA(:,1:3),cnTrainY');
figure;
plotResiduals(mdl)

% To plot true and predicted response
cnTrainXPCA = cnTrainX'*coeff;
cnTrainYPCA = predict(mdl,cnTrainXPCA(:,1:3));
figure;
plot(cnTrainY',cnTrainYPCA,'r*')
title('Linear regression model using PCA')
xlabel('True response');
ylabel('Predicted response');
axis([40 60 40 60])

% To plot residues
PCAresid = (cnTrainY'-cnTrainYPCA);
figure;
stem(cnTrainY, PCAresid)
title('Linear regression model using PCA')
xlabel('True response');
ylabel('Residual');
figure;
hist(PCAresid)
title('Linear regression model using PCA')
xlabel('Histogram of residuals');

% To predict the cetane numbers of cnTestX
cnTestXPCA = cnTestX'*coeff;
cnTestYPCA2 = predict(mdl,cnTestXPCA(:,1:3));

% To plot the predictions
figure;
plot(cnTestYPCA1, 'r*')
hold on;
plot(cnTestYPCA2, 'bo')
title('Linear regression model using PCA')
legend('PCA app','PCA manual','Location','north')
xlabel('Sample number')
ylabel('Cetane number')
xlim([0 113])
hold off;

% To print information about the data sets and the predictions
Min_value_cnTrainY = min(cnTrainY);
Max_value_cnTrainY = max(cnTrainY);
Mean_value_cnTrainY = mean(cnTrainY);
Min_value_cnTestY = min(cnTestYPCA1);
Max_value_cnTestY = max(cnTestYPCA1);
Mean_value_cnTestY = mean(cnTestYPCA1);


%% The PLS part
% To do the PLS regression
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(cnTrainX',cnTrainY', 5, 'cv', 5);

% To test the trained model model
cnTrainYPLS = [ones(size(cnTrainX',1),1), cnTrainX']*beta;

% To plot true and predicted response
figure;
plot(cnTrainY',cnTrainYPLS,'r*')
title('Partial least squares linear model')
xlabel('True response');
ylabel('Predicted response');
axis([40 60 40 60])

% To estimate the generalization error
TSS = sum((cnTrainY'-mean(cnTrainY')).^2);
RSS = sum((cnTrainY'-cnTrainYPLS).^2);
Rsquared = 1 - RSS/TSS;
newmse = (sum((cnTrainY'-cnTrainYPLS).^2))/133;
PLSresid = (cnTrainY'-cnTrainYPLS);

% To plot residues
figure;
stem(cnTrainY, PLSresid)
title('Partial least squares linear model')
xlabel('True response');
ylabel('Residual');
figure;
hist(PLSresid)
title('Partial least squares linear model')
xlabel('Histogram of residuals');

% To predict the cetane numbers of cnTestX
cnTestYPLS1 = [ones(size(cnTestX',1),1), cnTestX']*beta;

% To plot predictions from linear model with PCA and PLS
figure;
plot(cnTestYPCA1, 'r*')
hold on;
plot(cnTestYPCA2, 'bo')
plot(cnTestYPLS1, 'kx')
legend('PCA app','PCA manual','PLS','Location','north')
xlabel('Sample number')
ylabel('Cetane number')
xlim([0 113])
hold off;

% To plot validation and training errors of PLS model
for nn = 1:31
    [XL,yl,XS,YS,beta,PCTVAR,PLSValidationMSE,stats] = plsregress(cnTrainX',cnTrainY', nn, 'cv', 5);
    cnTrainSeries = [ones(size(cnTrainX',1),1), cnTrainX']*beta;
    PLSTrainMSE(nn) = (sum((cnTrainY'-cnTrainSeries).^2))/133;
end
nn = nn-1;
figure;
plot(1:nn,PLSValidationMSE(2,2:nn+1), 'ro-');
hold on;
plot(1:nn,PLSTrainMSE(1:nn), 'bo-');
title('Partial least squares linear model')
legend('Validation MSE','Training MSE')
xlabel('Number of PLS-components')
ylim([0 15])
hold off;

% Test
%{
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(cnTrainX',cnTrainY', 30, 'cv', 5);
figure;
plot(1:30,cumsum(100*PCTVAR(2,:)),'-bo');
title('Partial least squares linear model')
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
%}

%% The MLP part
% PCA
[coeff,score,latent,tsquared,explained] = pca(cnTrainX');
cnTrainXPCA = cnTrainX'*coeff;

% To test different numner of units and PCA-components
for jj = 1:20 % Units
    for kk = 1:10 % PCA-components
        [testmlp,tr] = train(feedforwardnet(jj),cnTrainXPCA(:,1:kk)',cnTrainY);
        cnTrainYMLP = testmlp(cnTrainXPCA(:,1:kk)');
        MLPValidationMSE(jj, kk)= tr.best_vperf;
    end
end

% To plot
figure;
surf(MLPValidationMSE)
title('MLP')
xlabel('Number of PCA-components')
ylabel('Number of units in the hidden layer')
zlabel('Validation MSE')

% Units and components to use
[M,I] = min(MLPValidationMSE(:));
[Units, Components] = ind2sub(size(MLPValidationMSE),I);
TestMLPMSE = M;
TestMPLRMSE = sqrt(M);

% To train the net
[mlp,tr] = train(feedforwardnet(10),cnTrainXPCA(:,1:5)',cnTrainY);
cnTrainYMLP = mlp(cnTrainXPCA(:,1:5)');
MLPMSE = tr.best_vperf;
MLPRMSE = sqrt(tr.best_vperf);

% To plot true and predicted response
figure;
plot(cnTrainY',cnTrainYMLP,'r*')
title('MLP')
xlabel('True response');
ylabel('Predicted response');
axis([40 60 40 60])

% To plot residues
MLPresid = (cnTrainY - cnTrainYMLP);
figure;
stem(cnTrainY, MLPresid)
title('MLP')
xlabel('True response');
ylabel('Residual');
figure;
hist(MLPresid)
title('MLP')
xlabel('Histogram of residuals');

% To predict
cnTestXPCA = cnTestX'*coeff;
cnTestYMLP1 = mlp(cnTestXPCA(:,1:5)');

% To plot predictions from linear model with PCA, PLS and MLP
figure;
plot(cnTestYPCA1, 'r*')
hold on;
plot(cnTestYPCA2, 'bo')
plot(cnTestYPLS1, 'kx')
plot(cnTestYMLP1, 'g+')
legend('PCA app','PCA manual','PLS','MLP','Location','north')
xlabel('Sample number')
ylabel('Cetane number')
xlim([0 113])
hold off;

