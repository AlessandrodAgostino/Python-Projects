train_prediction, y_cov = pipe_filter_gpr.predict(train_feats, return_cov = True)
  train_prediction = np.floor(train_prediction)
  train_score=pipe_filter_gpr.score(train_feats,y)

  data_train['predicted_age'] = train_prediction

  f=sns.lmplot('age_floor', 'predicted_age',data=data_train, robust=True,
              scatter_kws={'alpha':0.2}, height=8, ci=90)

  plt.gca().set_title(r'Final Model Full Train Result, $R^2=$%.2f'%train_score, size=15)
  plt.gca().set_ylabel('Predicted Age', size=15)
  plt.gca().set_xlabel('Age', size=15)
  f.savefig(pj(results_dir, 'Full_Train_final_Result_gender.png'), bbox_inches='tight')
