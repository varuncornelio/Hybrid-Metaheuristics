from EvoCluster import EvoCluster

optimizer = ["GWO"] #Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS".
objective_func = ["SSE", "TWCV"] #Select objective function from the list of available ones:"SSE","TWCV","SC","DB","DI".
dataset_list = ["iris"] #Select data sets from the list of available ones
num_of_runs = 3 #Select number of repetitions for each experiment. 
params = {'PopulationSize': 30, 'Iterations': 50} #Select general parameters for all optimizers (population size, number of iterations)
export_flags = {'Export_avg': True, 'Export_details': True, 'Export_details_labels': True,
                'Export_convergence': True, 'Export_boxplot': True} #Choose your preferemces of exporting files

ec = EvoCluster(
    optimizer,
    objective_func,
    dataset_list,
    num_of_runs,
    params,
    export_flags,
    auto_cluster=True,
    n_clusters='supervised',
    labels_exist=True,
    metric='euclidean'
)

ec.run() #run the framework