
alphas=np.arange(0.001, 10, 0.005)
lasso=LassoCV(alphas=alphas, fit_intercept=True, max_iter=100000)
ridge=RidgeCV(alphas=alphas, fit_intercept=True)
regressors=[lasso,ridge]
scalers=[RobustScaler(), StandardScaler(), MinMaxScaler()]

pipe_regr = Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', lasso)
                ])

param_grid = {'scaler': scalers, 'regressor': regressors}
cv=KF(10, shuffle=True)
find_regressor = GridSearchCV(  pipe_regr,
                                n_jobs=16,
                                pre_dispatch=8,
                                param_grid=param_grid,
                                cv=cv,
                                verbose=1))
find_regressor.fit(feats, y)
best_regressor = find_regressor.best_estimator_
alpha = best_regressor.alpha_
