# Countplot Visualization
def percent_countplot(df, col_name, col_hue = None, order_by_value = False, topn = 0, dropna = True,
                      orient = 'v', annot_percent = True, figsize = (14,8)):
    
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fill Paramter
    fs = 16 # Font size
    x_data = df.copy()
    x_col = col_name # column to count
    x_hue = col_hue # For divide column
    tittle = 'Distribution by '+x_col

    matplotlib.rc('xtick', labelsize = fs-2)     
    matplotlib.rc('ytick', labelsize = fs-2)
    fig = plt.figure(figsize= figsize)
    
    data_col = x_data   
        
    # topn > 0 we will combine the rest as "Other"
    if topn:
        counts = data_col[x_col].value_counts()
        topn_name = counts.nlargest(topn).index
        data_col[x_col] = data_col[x_col].where(data_col[x_col].isin(topn_name), other = 'Other')
    elif topn == 0:
        topn = data_col[x_col].nunique()
        
    # arrange order by value or index?
    if order_by_value :
        counts = data_col[x_col].value_counts()
        max_count = max(counts) # for adjust text alignment
    else:
        if str in list(map(type,data_col[x_col])) or not dropna:
            data_col[x_col] = data_col[x_col].astype(str)
        counts = data_col[x_col].value_counts().sort_index()
        max_count = max(counts.iloc[:topn]) # for adjust text alignment
        
    if orient == 'v':
        ax = sns.countplot(x = x_col, hue = x_hue, data= data_col, order = counts.iloc[:topn+1].index)
        ax.set_xlabel(x_col, color = 'r', fontsize = fs, fontweight='bold')
        ax.set_ylabel('Frequency', color = 'b', fontsize = fs, fontweight='bold')
        # Set rotation of xticks if name is too long
        data_col[x_col] = data_col[x_col].astype(str)
        xrot = 15 if max(list(map(len,data_col[x_col].unique()))) > 10 else 0
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    elif orient == 'h':
        ax = sns.countplot(y = x_col, hue = x_hue, data= data_col, order = counts.iloc[:topn+1].index)
        ax.set_ylabel(x_col, color = 'b', fontsize = fs, fontweight='bold')
        ax.set_xlabel('Frequency', color = 'r', fontsize = fs, fontweight='bold')
        xrot = 0
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        
    total_count = sum(counts) # for calculate percentage
    # print(counts)
    
    # fig.suptitle('test title', fontsize=12)
    ax.set_title(tittle, fontsize = fs, fontweight='bold' )

    plt.xticks(rotation=xrot, color='r', size=16)
    plt.yticks(rotation=0, color='b', size=16)
    
    if x_hue == None and orient == 'v':
        for i, v in enumerate(counts[:topn+1]):
            ax.text(x = i, y=v + max_count*0.01 , s='{:,}'.format(v), horizontalalignment='center', color='black', fontweight='bold')
            if annot_percent:
                ax.text(x = i, y= v/2, s=str('{:.2f}'.format(v*100/total_count))+'%',
                        color='yellow', fontweight='bold', size = 14,
                        horizontalalignment='center', 
                        verticalalignment='center' 
                        )
    elif x_hue == None and orient == 'h':
        for i, v in enumerate(counts[:topn+1]):
            ax.text(x = v + max_count*0.03, y=i , s='{:,}'.format(v), horizontalalignment='center', color='black', fontweight='bold')
            if annot_percent:
                ax.text(x = v/2, y=i , s=str('{:.2f}'.format(v*100/total_count))+'%',
                        color='yellow', fontweight='bold', size = 14,
                        horizontalalignment='center', 
                        verticalalignment='center' 
                        )
    return ax

def chi2_indtest(df_feature, df_target, pvalue = 0.05, verbose =0):
    import pandas as pd
    from scipy.stats import chi2_contingency
    feature_list_chi = []
    feature_list_chi_score = []
    for series in df_feature:
        nl = "\n"

        crosstab = pd.crosstab(df_feature[series], df_target.values.ravel())
        if verbose: print(crosstab, nl)
        chi2, p, dof, expected = chi2_contingency(crosstab)
        if verbose: print(f"Chi2 value= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}{nl}")
        if p < pvalue:
            feature_list_chi.append(series)
            feature_list_chi_score.append(chi2)
    return feature_list_chi, feature_list_chi_score

def plot_feature_importances(df, threshold = 0.90, normalized = True):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index(drop=True)
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    colors = ['b', 'g', 'r', 'c', 'm', 'y','#4ef7ae','#d96d09', '#2b9900','#f7184d', '#1b5c44','#f25e60','#e59400']

    if normalized:
        ax.barh(list(reversed(list(df.index[:15]))), 
                df['importance_normalized'].head(15), 
                align = 'center', edgecolor = 'k',color = colors)
        plt.xlabel('Normalized Importance')
    else:
        ax.barh(list(reversed(list(df.index[:15]))), 
                df['importance'].head(15), 
                align = 'center', edgecolor = 'k',color = colors)
        plt.xlabel('Importance')
        # Set the xticks format
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.2f}".format(int(x))))
        
#     for i, v in enumerate(df['importance_normalized'].head(15)):
#         ax.text(v + 0.001, i , '{:.4f}'.format(v), color='blue', fontweight='bold')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    
    
    # Plot labeling
    plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    n_fea = len(df)
    ax.plot(np.arange(n_fea)+1, df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    
    if threshold:
        # Index of minimum number of features needed for cumulative importance threshold
        # np.where returns the index so need to add 1 to have correct number
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        plt.vlines(x = importance_index + 1, ymin = 0, ymax = threshold, 
                   linestyles='--', colors = 'blue' )
        plt.text(importance_index +0.02*n_fea, 0, str(importance_index + 1), color='red', fontweight='bold')
        plt.text(importance_index +0.05*n_fea, threshold, str(threshold*100)+'%', color='orange', fontweight='bold')
        plt.show();

    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    
    return df


# Before we use 'get_dummies' function we have to convert data type of all feature to be 'category' 
def cvt_type(df, col_list, dtype = 'category'):
    for col in col_list:
        df[col] = df[col].astype(int).astype('category')
    return df

# Oversampling
def makeOverSamplesADASYN(X,y):
    from imblearn.over_sampling import ADASYN
    """
    Purpose
    ----------
    Increasing the observation of minority class 

    Parameters
    ----------
    X: Independent Variable in DataFrame
    y: Dependent Variable in Pandas DataFrame format
    Returns:
    ----------
    Returns Independent and Dependent variable with resampling minority class
    """
    X_resampled, y_resampled = ADASYN(random_state=7).fit_sample(X, y)
    return(X_resampled, y_resampled)


# Logistic Regression
def LogReg_HPTune(X, y, verbose = 0):
    """
    Purpose
    ----------
    Choosing a set of optimal hyperparameters for a Logistic Regression Classifier

    Parameters
    ----------
    X: Data set with all feature or predictor
    y: Data set with Class
    verbose: 0 mean not show summary of tuning
             1 mean show summary of tuning
    Returns:
    ----------
    Returns grid search model of Logistic Regression Classifier with tuned hyperparameter
    """  
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    # Logistic Regression Classifier
    LogReg_clf = LogisticRegression(random_state = 7, fit_intercept=False)
    
    # Create regularization hyperparameter space
    C = np.logspace(-5, 5, 100)
    
    # Create regularization penalty space
    penalty = ['none', 'l2']
    
    # Create class weight mode space
    class_weight = [None, 'balanced']
    
    # Create solver function space
    solver = ['saga', 'lbfgs', 'newton-cg']

    
    # Define Search Param
    param_dist = dict(C = C,
                      penalty = penalty,
                      class_weight = class_weight,
#                       solver = solver
                     )
    rs = RandomizedSearchCV(estimator=LogReg_clf,
                            param_distributions=param_dist,
                            refit=True,
                            scoring=score_param,
                            n_iter=n_iter_search,
                            cv=cv,
                            n_jobs=-1,
                            verbose =1,
                            random_state=7,
                            iid=True)
    if verbose == 1:
        start = time()    
        rs.fit(X,y)
        print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." 
              % (time() - start, len(rs.cv_results_['params'])))

    elif verbose == 0:
        rs.fit(X,y)
    
    # Best parameter from RandomizedSearchCV
    bs_C = rs.best_params_['C'] 
    bs_penalty = rs.best_params_['penalty'] 
    bs_class_weight = rs.best_params_['class_weight']
#     bs_solver = rs.best_params_['solver'] 
    d_C = np.log10(bs_C)
    
    param_grid = dict(C = np.append(bs_C,np.logspace(d_C-2,d_C+2,num=100)),
                      penalty = [bs_penalty],
                      class_weight = [bs_class_weight],
#                       solver = [bs_solver]
                     )
    gs = GridSearchCV(estimator=LogReg_clf, 
                      param_grid=param_grid,
                      refit=True,
                      scoring=score_param,
                      cv=cv,
                      n_jobs=-1,
                      verbose =1,
                      iid=True)
    
    if verbose == 1:
        start = time()
        gs.fit(X,y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))

    elif verbose == 0:
        gs.fit(X,y)
    
    return rs, gs


# Logistic Regression with ovesampling
def LogReg_HPTune_w_OverSam(X, y, verbose = 0):
    """
    Purpose
    ----------
    Choosing a set of optimal hyperparameters for a Logistic Regression Classifier with balancing data

    Parameters
    ----------
    X: Data set with all feature or predictor
    y: Data set with Class
    verbose: 0 mean not show summary of tuning
             1 mean show summary of tuning
    Returns:
    ----------
    Returns grid search model of Logistic Regression Classifier with tuned hyperparameter
    """  
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    # Oversampling
    X, y = makeOverSamplesADASYN(X,y)
    
    # Logistic Regression Classifier
    LogReg_clf = LogisticRegression(random_state = 7, max_iter=1000, fit_intercept=False)
    
    # Create regularization hyperparameter space
    C = [0.1,1,10] #np.logspace(-1, 1, 50)
    
    # Create regularization penalty space
    penalty = ['none', 'l2']
    
    # Create class weight mode space
    class_weight = [None, 'balanced']
    
    # Create solver function space
    solver = ['saga', 'lbfgs', 'newton-cg']
    
    # 
    fit_intercept = [True, False]

    
    # Define Search Param
    param_dist = dict(C = C,
#                       penalty = penalty,
#                       class_weight = class_weight,
#                       solver = solver,
#                       fit_intercept = fit_intercept,
                     )
    rs = RandomizedSearchCV(estimator=LogReg_clf,
                            param_distributions=param_dist,
                            refit=True,
                            scoring=score_param,
                            n_iter=n_iter_search,
                            cv=cv,
                            n_jobs=-1,
                            verbose =verbose,
                            random_state=7,
                            iid=True)
    if verbose == 1:
        start = time()    
        rs.fit(X,y)
        print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." 
              % (time() - start, len(rs.cv_results_['params'])))

    elif verbose == 0:
        rs.fit(X,y)
    
    # Best parameter from RandomizedSearchCV
    bs_C = rs.best_params_['C'] 
#     bs_penalty = rs.best_params_['penalty'] 
#     bs_class_weight = rs.best_params_['class_weight']
#     bs_solver = rs.best_params_['solver'] 
#     bs_fit_intercept = rs.best_params_['fit_intercept']
    d_C = np.log10(bs_C)
    
    param_grid = dict(C = np.append(bs_C,np.logspace(d_C-4,d_C+4,num=50)),
#                       penalty = [bs_penalty],
#                       class_weight = [bs_class_weight],
#                       solver = [bs_solver],
                      fit_intercept = fit_intercept #[bs_fit_intercept],
                     )
    gs = GridSearchCV(estimator=LogReg_clf, 
                      param_grid=param_grid,
                      refit=True,
                      scoring=score_param,
                      cv=cv,
                      n_jobs=-1,
                      verbose =verbose,
                      iid=True)
    
    if verbose == 1:
        start = time()
        gs.fit(X,y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))

    elif verbose == 0:
        gs.fit(X,y)
    
    return rs, gs

# Naive Bayes
def NB_HPTune(X, y, verbose = 0):
    """
    Purpose
    ----------
    Choosing a set of optimal hyperparameters for a Naive Bayes Classifier 

    Parameters
    ----------
    X: Data set with all feature or predictor
    y: Data set with Class
    verbose: 0 mean not show summary of tuning
             1 mean show summary of tuning
    Returns:
    ----------
    Returns grid search model of Naive Bayes Classifier with tuned hyperparameter
    """  
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    gnb = GaussianNB(priors=None)
    param_dist = dict(var_smoothing = np.logspace(-16,0,200) ) # default is 1e-9
    rs = RandomizedSearchCV(estimator=gnb,
                             param_distributions=param_dist,
                             scoring=score_param,
                             refit=True,
                             n_iter = n_iter_search,
                             cv=cv,
                             n_jobs=-1,
                             random_state=7,
                             iid=True)
    if verbose == 1:
        start = time()    
        rs.fit(X, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." 
              % (time() - start, len(rs.cv_results_['params'])))
    elif verbose == 0:
        rs.fit(X, y)
    
    # Best parameter from RandomizedSearchCV
    bs_var_sm = rs.best_params_['var_smoothing']
    bs_var_sm_pw = np.log10(bs_var_sm)
    
    param_grid = dict(var_smoothing = np.logspace(bs_var_sm_pw*0.9,bs_var_sm_pw*1.1,50))
    gs = GridSearchCV(estimator=gnb, 
                      param_grid=param_grid,
                      scoring=score_param,
                      refit=True,
                      cv=cv,
                      n_jobs=-1, 
                      iid=True)
    
    if verbose == 1:
        start = time()
        gs.fit(X, y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))
    elif verbose == 0:
        gs.fit(X, y)
    
    return rs, gs

# Semi-supervised learning: Pseudo-labeling
def Pseudo_labeling(df_train, df_NaN, LogReg_HPTune_w_OverSam, n_sam = 100, verbose=0):
    
    print("Initiate base model from training set....")
    # Training set
    X_encoded = df_train.drop(['Drug_Rechallenge_Result','Patient_ID'], axis=1)
    y_data = df_train[['Drug_Rechallenge_Result']]
    
    print(X_encoded.shape)
    print(y_data.shape)
    
    score_param = 'accuracy' # Score for tune model
    n_iter_search = 50 # Max candidate parameter for RandomizedSearchCV
    cv = 5 # Number of k-fold cross validation
    LogReg_rs, LogReg_gs = LogReg_HPTune_w_OverSam(X_encoded,y_data.values.ravel(), verbose=0)
    
    # Make a copy of df_NaN
    df_NaN_temp = df_NaN.copy()
    
    n_sam = n_sam
    n_min = n_sam
    n_frac = n_sam
    if type(n_sam) != int:
        n_sam = n_frac * len(y_data)
        n_min = 100
    n_round = 0
    
    print("Start Active Learning Process")
    
    while len(df_NaN_temp) > n_min and len(df_NaN_temp) > n_sam:
        n_round += 1
        print("Round: ", n_round, "\nProcessing....")
        
        # Sampling n_sam sample to predict class and store idx with high confidence
        if type(n_frac) != int:
            n_sam = n_frac * len(y_data)
        df_NaN_train = df_NaN_temp.sample(n = int(n_sam), random_state=7)
        df_NaN_temp = df_NaN_temp.drop(df_NaN_train.index)
        
        X_encoded_NaN = df_NaN_train.drop(['Drug_Rechallenge_Result','Patient_ID'], axis=1)
#         y_data_NaN = df_NaN_train[['Drug_Rechallenge_Result']]

        # Search idx with high predict_proba
        high_confidence_idx = []
        predict_result = []
        threshold_pos = 0.8
        for idx in X_encoded_NaN.index:
            Pos_confidence = LogReg_gs.predict_proba(X_encoded_NaN.loc[[idx]])[:,1]
            if Pos_confidence > threshold_pos:
                high_confidence_idx.append(idx)
                predict_class = LogReg_gs.predict(X_encoded_NaN.loc[[idx]])
                predict_result.append(predict_class)
            elif Pos_confidence < (1-threshold_pos):
                high_confidence_idx.append(idx)
                predict_class = LogReg_gs.predict(X_encoded_NaN.loc[[idx]])
                predict_result.append(predict_class)

        # Create dataframe to store high confidence result
        df_feature_NaN = X_encoded_NaN.loc[high_confidence_idx]
        df_predict_NaN = pd.DataFrame(predict_result, columns=['Drug_Rechallenge_Result'], index=high_confidence_idx)

        # Concat new predicted data to X_encoded and y_data
        X_encoded = pd.concat([X_encoded,df_feature_NaN])
        y_data = pd.concat([y_data,df_predict_NaN])
        
        print(X_encoded.shape)
        print(y_data.shape)

        # GridSearch CV - Retraining
        score_param = 'accuracy' # Score for tune model
        n_iter_search = 100 # Max candidate parameter for RandomizedSearchCV
        cv = 5 # Number of k-fold cross validation
        LogReg_rs, LogReg_gs = LogReg_HPTune_w_OverSam(X_encoded,y_data.values.ravel(), verbose=0)

        # Append row which has low confident to df_NaN_temp
        low_con_idx = df_NaN_train[~df_NaN_train.index.isin(high_confidence_idx)].index
        df_NaN_temp = pd.concat([df_NaN_temp,df_NaN_train.loc[low_con_idx]])
        
        # Display performance of each round
        print('Training score: ', LogReg_rs.best_score_)
        print('Test score: ', accuracy_score(LogReg_rs.predict(X_encoded_test),y_data_test))
        
    # If the rest NaN sample less than n_sam
    df_NaN_train = df_NaN_temp
    X_encoded_NaN = df_NaN_train.drop(['Drug_Rechallenge_Result','Patient_ID'], axis=1)
    predict_class = LogReg_gs.predict(X_encoded_NaN)
    df_predict_NaN = pd.DataFrame(predict_class, columns=['Drug_Rechallenge_Result'], index=X_encoded_NaN.index)
    
    # Concat new predicted data to X_encoded and y_data
    X_encoded = pd.concat([X_encoded,X_encoded_NaN])
    y_data = pd.concat([y_data,df_predict_NaN])   
    
    print("=========== Finish ===========")
    print(X_encoded.shape)
    print(y_data.shape)
    
    return X_encoded, y_data, LogReg_gs

