
#inspired by:
#https://stackoverflow.com/questions/4622057/plotting-3d-polygons-in-python-matplotlib

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pandas as pd #for confusion matrix 
import seaborn as sns #for confusion matrix 
import numpy as np
from sklearn import metrics


def plot_latent_trajectory(path_to_save_fig=None):
	fig = plt.figure()
	ax = Axes3D(fig)

	#vertices of simplex
	x = [0,0,1]
	y = [0,1,0]
	z = [1,0,0]
	verts = [zip(x,y,z)]


	#trajectory I made up
	xx=[[0.35, 0.2], [0.2, 0.32], [0.32, 0.25], [0.25, 0.1] ]
	yy=[[0.55, 0.5], [0.5,0.44],  [0.44,0.50], [0.50, 0.1]]
	zz=[[0.10, 0.1], [0.1, 0.16], [0.16, 0.25], [0.25, 0.8]]

	p=Poly3DCollection(verts, alpha=0.2)
	p.set_edgecolor('#000000')
	p.set_facecolor('#00FFFF')
	ax.add_collection3d(p)
	ax.view_init(30, 240)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	ax.invert_xaxis()
	ax.invert_yaxis()

	#make the simplex (plot each point + its index as text above)
	for i in range(len(x)): 
	 ax.scatter(x[i],y[i],z[i] ,color='b', s=50) 
	 ax.text(x[i],y[i],z[i], '(%d,%d,%d)' % (x[i],y[i],z[i]), size=10, zorder=1,  
	 color='k') 
	#add in trajectory
	for (x,y,z) in zip(xx,yy,zz):
		ax.plot(x,y,z, 'k-o')
	#save or plot
	if path_to_save_fig:
		fig.savefig(path_to_save_fig)
	else:
		fig.show()

def create_confusion_matrix_plot(ys, ys_predicted, legend=True, path_to_save_fig=None):
	#inspired by bacon 
	labels = list(set(ys).union(set(ys_predicted)))
	cm = metrics.confusion_matrix(ys, ys_predicted, labels=labels)

	plt.clf()
	plt.figure(figsize=(25, 20))
	df_cm = pd.DataFrame(cm, index=labels)
	ax = sns.heatmap(df_cm, annot=True, linewidths=.5, cmap="OrRd", cbar=legend)
	ax.set(xlabel='Predicted', ylabel='True')
	ax.set_xticklabels(labels, rotation=90)
	plt.yticks(rotation=0)
	if path_to_save_fig:
		plt.savefig(path_to_save_fig)

def plot_accuracies(accs_model, accs_baserate, method1_name="Generic deobfuscator",
		 method2_name="Baserate", title="",  ylim=None, path_to_save_fig=None):
	"""
	Generate a simple plot of the test and training learning curve.

	Inspired by:
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

	Parameters
	----------
	ylim : tuple, shape (ymin, ymax), optional
		Defines minimum and maximum yvalues plotted.

	"""
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Number of predictions")
	plt.ylabel("Frequency of including true API call")
	plt.grid()

	N=len(accs_model)
	x_vals=[x+1 for x in range(N)]
	plt.xticks(np.arange(min(x_vals), max(x_vals)+1, 1.0))

	plt.plot(x_vals, accs_model, 'o-', color="g",
			 label=method1_name)
	plt.plot(x_vals, accs_baserate, 'o--', color="r",
			 label=method2_name)

	plt.legend(loc="best")
	if path_to_save_fig:
		plt.savefig(path_to_save_fig)
	return plt

	#api_functions=dict_api_calls_to_n_args.keys()
