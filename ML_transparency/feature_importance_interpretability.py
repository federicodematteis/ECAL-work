#questo script serve come draft per interpretare le scelte dei pesi del modello, e capire quale feature è la più importante nella scelta di questi ultimi.
#serve fare il load di uno dei modelli addestrati e le funzioni di data preparation per compilare questo script.


#plot of the model 
import visualkeras

visualkeras.layered_view(saved_model, to_file=f'{data_folder}/DNNPLOT.png', min_xy=10, min_z=10, scale_xy=100, scale_z=100, one_dim_orientation='x')

#explainability tools for machine learning 

import lime
import lime.lime_tabular
from sklearn import linear_model

#set up the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(all_inputs_validation,
                                                  training_labels = None,
                                                  feature_names = ['lumi_inst', 'lumi_in_fill', 'lumi_LHC', 'time_in_fill', 'lumi_last_fill', 'lumi_last_point'],
                                                  #feature_selection='lasso_path',
                                                  mode = 'regression',
                                                  discretize_continuous = False)

# you need to modify the output since keras outputs a tensor and LIME takes arrays
def predict(x):
    return saved_model.predict(x).flatten()

# compute the explainer. Chose Huber for its robustness against outliers
i=100
exp = explainer.explain_instance(all_inputs_validation[i,:],
                                  predict,
                                  num_features=6,
                                  distance_metric='euclidean',
                                 num_samples=len(all_inputs_validation),
                                 model_regressor = linear_model.HuberRegressor())

# generate plot for one item
exp.show_in_notebook(show_table=True, predict_proba=True, show_predicted_value=True)
[exp.as_pyplot_figure(label=1)]
#plt.figure()
plt.show()

#SP LIME 
from lime import submodular_pick
#set up sp lime with 20 samples. The more amount of samples time increases dramatically
sp_obj = submodular_pick.SubmodularPick(explainer, 
                                        all_inputs_validation,
                                        predict, 
                                        sample_size=len(all_inputs_validation),
                                        num_features=6,
                                        num_exps_desired=5)

#get explanation matrix
W_matrix = pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])

#get overall mean explanation for each feature
matrix_mean = W_matrix.mean()
plt.figure(figsize=(6,6))

matrix_mean.sort_values(ascending=False).plot.bar()
plt.show()

