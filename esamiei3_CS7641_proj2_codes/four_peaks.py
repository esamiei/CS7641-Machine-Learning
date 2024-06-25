"""
Student Name: Ehsan Samiei  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: esamiei3  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903952270   		  	   		  		 		  		  		    	 		 		   		 		  
"""  


import numpy as np 
import mlrose_hiive
import matplotlib.pyplot as plt
import time

np.random.seed(903952270)   # student ID

### We start with tuning the hyperparameters. four algorithm four hyperparameter in the following four functions will be tuned. 

def four_peaks_fitness_curve_hyperparameter_sa(t_pct_threshold=0.15, state_length=200):

    fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
    problem = mlrose_hiive.DiscreteOpt(length=state_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=state_length)

    _, _, fitness_curv_sa_1 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(), max_attempts= 20, max_iters=10000, init_state=init_state, curve=True)
    _, _, fitness_curv_sa_2 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(), max_attempts= 20, max_iters=10000, init_state=init_state, curve=True)
    _, _, fitness_curv_sa_3 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ArithDecay(), max_attempts= 20, max_iters=10000, init_state=init_state, curve=True)

    gen_plot = True
    if gen_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("SA Analysis using 'schedule' as the hyperparameter")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        plt.plot(fitness_curv_sa_1[:, 0], label='Exponential')
        plt.plot(fitness_curv_sa_2[:, 0], label='Geometric')
        plt.plot(fitness_curv_sa_3[:, 0], label='Arithmetic')
        plt.grid()
        plt.legend(loc='best')
        fig.savefig("Images/four_peaks_sa_hyperparameter.png")
        #plt.show()

    best_schedule_idx= np.argmax([[fitness_curv_sa_1[-1,0]],[fitness_curv_sa_2[-1,0]],[fitness_curv_sa_3[-1,0]]])
    best_schedule=best_schedule_idx
    print("fitness review based on schedule hyperparameter for SA completed with best at->",best_schedule)
    
    return best_schedule

def four_peaks_fitness_curve_hyperparameter_rhc(t_pct_threshold=0.15, state_length=200):

    fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
    problem = mlrose_hiive.DiscreteOpt(length=state_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=state_length)

    restarts = [0, 5, 10, 20, 40]
    _, _, fitness_curv_rhc_1 = mlrose_hiive.random_hill_climb(problem, restarts=restarts[0], max_attempts= 20, max_iters=10000, init_state=init_state, curve=True)
    _, _, fitness_curv_rhc_2 = mlrose_hiive.random_hill_climb(problem, restarts=restarts[1], max_attempts= 20, max_iters=10000,init_state=init_state, curve=True)
    _, _, fitness_curv_rhc_3 = mlrose_hiive.random_hill_climb(problem, restarts=restarts[2], max_attempts= 20, max_iters=10000,init_state=init_state, curve=True)
    _, _, fitness_curv_rhc_4 = mlrose_hiive.random_hill_climb(problem, restarts=restarts[3], max_attempts= 20, max_iters=10000,init_state=init_state, curve=True)
    _, _, fitness_curv_rhc_5 = mlrose_hiive.random_hill_climb(problem, restarts=restarts[4], max_attempts= 20, max_iters=10000,init_state=init_state, curve=True)

    gen_plot = True
    if gen_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("RHC Analysis using 'Restarts' as the hyperparameter")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        plt.plot(fitness_curv_rhc_1[:, 0], label='restarts= {}'.format(restarts[0]))
        plt.plot(fitness_curv_rhc_2[:, 0], label='restarts= {}'.format(restarts[1]))
        plt.plot(fitness_curv_rhc_3[:, 0], label='restarts= {}'.format(restarts[2]))
        plt.plot(fitness_curv_rhc_4[:, 0], label='restarts= {}'.format(restarts[3]))
        plt.plot(fitness_curv_rhc_5[:, 0], label='restarts= {}'.format(restarts[4]))
        plt.grid()
        plt.legend(loc='best')
        fig.savefig("Images/four_peaks_rhc_hyperparameter.png")
        #plt.show()

    best_restarts_idx= np.argmax([[fitness_curv_rhc_1[-1,0]],[fitness_curv_rhc_2[-1,0]],[fitness_curv_rhc_3[-1,0]],[fitness_curv_rhc_4[-1,0]]])
    best_restarts=restarts[best_restarts_idx]
    print("fitness review based on restarts hyperparameter for RHC completed with best at->",best_restarts)
    
    return best_restarts


def four_peaks_fitness_curve_hyperparameter_ga(t_pct_threshold=0.15, state_length=200):

    fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
    problem = mlrose_hiive.DiscreteOpt(length=state_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=state_length)

    mutation_prob = [0.2, 0.4, 0.6, 0.8]
    _, _, fitness_curv_ga_1 = mlrose_hiive.genetic_alg(problem, mutation_prob=mutation_prob[0], max_attempts=20, max_iters=10000, curve=True)
    _, _, fitness_curv_ga_2 = mlrose_hiive.genetic_alg(problem, mutation_prob=mutation_prob[1], max_attempts=20, max_iters=10000, curve=True)
    _, _, fitness_curv_ga_3 = mlrose_hiive.genetic_alg(problem, mutation_prob=mutation_prob[2], max_attempts=20, max_iters=10000, curve=True)
    _, _, fitness_curv_ga_4 = mlrose_hiive.genetic_alg(problem, mutation_prob=mutation_prob[3], max_attempts=20, max_iters=10000, curve=True)

    gen_plot = True
    if gen_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("GA Analysis using 'Mutation Probability' as the hyperparameter")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        plt.plot(fitness_curv_ga_1[:, 0], label='mutation_prob= {}'.format(mutation_prob[0]))
        plt.plot(fitness_curv_ga_2[:, 0], label='mutation_prob= {}'.format(mutation_prob[1]))
        plt.plot(fitness_curv_ga_3[:, 0], label='mutation_prob= {}'.format(mutation_prob[2]))
        plt.plot(fitness_curv_ga_4[:, 0], label='mutation_prob= {}'.format(mutation_prob[3]))
        plt.grid()
        plt.legend(loc='best')
        fig.savefig("Images/four_peaks_ga_hyperparameter.png")
        #plt.show()

    best_mutation_prob_idx= np.argmax([[fitness_curv_ga_1[-1,0]],[fitness_curv_ga_2[-1,0]],[fitness_curv_ga_3[-1,0]],[fitness_curv_ga_4[-1,0]]])
    best_mutation_prob=mutation_prob[best_mutation_prob_idx]
    print("fitness review based on mutation_prob hyperparameter for GA completed with best at->",best_mutation_prob)
    
    return best_mutation_prob

def four_peaks_fitness_curve_hyperparameter_mimic(t_pct_threshold=0.15, state_length=200):

    fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
    problem = mlrose_hiive.DiscreteOpt(length=state_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=state_length)

    pop_size = [200,300,400,500]
    _, _, fitness_curv_mimic_1 = mlrose_hiive.mimic(problem, pop_size=pop_size[0], max_attempts= 20, max_iters=10000, curve=True)
    _, _, fitness_curv_mimic_2 = mlrose_hiive.mimic(problem, pop_size=pop_size[1], max_attempts= 20, max_iters=10000, curve=True)
    _, _, fitness_curv_mimic_3 = mlrose_hiive.mimic(problem, pop_size=pop_size[2], max_attempts= 20, max_iters=10000, curve=True)
    _, _, fitness_curv_mimic_4 = mlrose_hiive.mimic(problem, pop_size=pop_size[3], max_attempts= 20, max_iters=10000, curve=True)

    gen_plot = True
    if gen_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Mimic Analysis using 'pop_size' as the hyperparameter")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        plt.plot(fitness_curv_mimic_1[:, 0], label='pop_size= {}'.format(pop_size[0]))
        plt.plot(fitness_curv_mimic_2[:, 0], label='pop_size= {}'.format(pop_size[1]))
        plt.plot(fitness_curv_mimic_3[:, 0], label='pop_size= {}'.format(pop_size[2]))
        plt.plot(fitness_curv_mimic_4[:, 0], label='pop_size= {}'.format(pop_size[3]))
        plt.grid()
        plt.legend(loc='best')
        fig.savefig("Images/four_peaks_mimic_hyperparameter.png")
        
    best_pop_size_idx= np.argmax([[fitness_curv_mimic_1[-1,0]],[fitness_curv_mimic_2[-1,0]],[fitness_curv_mimic_3[-1,0]],[fitness_curv_mimic_4[-1,0]]])
    best_pop_size=pop_size[best_pop_size_idx]
    print("fitness review based on pop_size hyperparameter for MIMIC completed with best at->",best_pop_size)
    
    return best_pop_size

# Now that the optimization algorithms tune, we can run them through iterations to review the accuracy, performance, and run time.

def     four_peaks_fitness_curve_iterations(best_schedule,best_restarts,best_mutation_prob,best_pop_size,t_pct_threshold=0.15, state_length=200):

    fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
    problem = mlrose_hiive.DiscreteOpt(length=state_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=state_length)
    if best_schedule ==2:
        best_schedule = mlrose_hiive.ArithDecay()
    elif best_schedule ==1:
        best_schedule =mlrose_hiive.GeomDecay()
    else:
        best_schedule = mlrose_hiive.ExpDecay()    

    start = time.time()
    _, _, fitness_curv_sa = mlrose_hiive.simulated_annealing(problem, schedule=best_schedule, max_attempts= 20, max_iters=10000, init_state=init_state, curve=True)
    end = time.time()
    sa_run_time=end-start
    
    start = time.time()
    _, _, fitness_curv_rhc = mlrose_hiive.random_hill_climb(problem,restarts=best_restarts, max_attempts= 20, max_iters=10000, init_state = init_state, curve=True)
    end = time.time()
    rhc_run_time=end-start
    
    start = time.time()
    _, _, fitness_curv_ga = mlrose_hiive.genetic_alg(problem, mutation_prob=best_mutation_prob, max_attempts= 20, max_iters=10000, curve=True)
    end = time.time()
    ga_run_time=end-start
    
    start = time.time()
    _, _, fitness_curv_mimic = mlrose_hiive.mimic(problem, pop_size=best_pop_size, max_attempts= 20, max_iters=10000, curve=True)
    end = time.time()
    mimic_run_time=end-start
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Fitness Curve vs number of Iterations | Four Peak Problem")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('fitness')
    ax.plot(fitness_curv_sa[:, 0], label='Simulated Annealing')
    ax.plot(fitness_curv_rhc[:, 0], label='Randomized Hill Climb')
    ax.plot(fitness_curv_ga[:, 0], label='Genetic Algorithm')
    ax.plot(fitness_curv_mimic[:, 0], label='MIMIC')
    ax.grid()
    ax.legend(loc='best')
    fig.savefig("Images/four_peaks_iterations.png")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Best fitness score | FourPeaks")
    ax.set_xlabel('Optimization Algorithm')
    ax.set_ylabel('Best fitness score')
    values=[fitness_curv_sa[-1,0], fitness_curv_rhc[-1,0], fitness_curv_ga[-1,0], fitness_curv_mimic[-1,0]]
    plt.bar(['SA','RHC', 'GA', 'MIMIC'],values)
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, str(value), ha='center')    
    fig.savefig("Images/four_peaks_iteration_best_score.png")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Run tim| FourPeaks")
    ax.set_xlabel('Optimization Algorithm')
    ax.set_ylabel('Run time')
    values=[round(sa_run_time,2), round(rhc_run_time,2), round(ga_run_time,2), round(mimic_run_time,2)]
    plt.bar(['SA','RHC', 'GA', 'MIMIC'],values)
    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha='center')    
    fig.savefig("Images/four_peaks_iteration_run_time.png")
    

    print("fitness review based on iterations completed")

### Additional work: review of the best tuned alogrithms with different n-dimensional state size 

def four_peaks_state_size(best_schedule,best_restarts,best_mutation_prob,best_pop_size,t_pct_threshold=0.15, state_size_tunned_for=200, length=range(5, 206, 20)):

    best_fitness = np.empty((len(length), 4))  # four columns represents SA,RHC,GA,MIMIC, respectively
    performance_time = np.empty((len(length), 4))
    state_length = np.empty(len(length))
    i = 0

    for val in length:
        fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
        problem = mlrose_hiive.DiscreteOpt(length=val, fitness_fn=fitness, maximize=True, max_val=2)
        problem.set_mimic_fast_mode(True)
        if best_schedule ==2:
            best_schedule = mlrose_hiive.ArithDecay()
        elif best_schedule ==1:
            best_schedule =mlrose_hiive.GeomDecay()
        else:
            best_schedule = mlrose_hiive.ExpDecay()   
        init_state = np.random.randint(2, size=val)
        state_length[i] = val
        
        start = time.time()
        _, best_fitness[i, 0], _ = mlrose_hiive.simulated_annealing(problem, schedule=best_schedule, max_attempts=20, max_iters=10000, init_state=init_state, curve=True)
        end = time.time()
        performance_time[i, 0] = end-start


        start = time.time()
        _, best_fitness[i, 1], _ = mlrose_hiive.random_hill_climb(problem, restarts= best_restarts, max_attempts=20, max_iters=10000, init_state=init_state, curve=True)
        end = time.time()
        performance_time[i, 1] = end-start


        start = time.time()
        _, best_fitness[i, 2], _ = mlrose_hiive.genetic_alg(problem, mutation_prob=best_mutation_prob, max_attempts=20, max_iters=10000, curve=True)
        end = time.time()
        performance_time[i, 2] = end-start


        start = time.time()
        _, best_fitness[i, 3], _ = mlrose_hiive.mimic(problem, pop_size=best_pop_size, max_attempts=20, max_iters=10000, curve=True)
        end = time.time()
        performance_time[i, 3] = end-start

        i = i+1

    gen_plot = True

    if gen_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Fitness function vs n-dimensional state space with T=15%")
        ax.set_xlabel('n-dimensional state space')
        ax.set_ylabel('fitness at the best state')
        plt.plot(state_length, best_fitness[:, 0], label='Simulated Annealing')
        plt.plot(state_length, best_fitness[:, 1], label='Randomized Hill Climb')
        plt.plot(state_length, best_fitness[:, 2], label='Genetic Algorithm')
        plt.plot(state_length, best_fitness[:, 3], label='MIMIC')
        ax.axvline(x=state_size_tunned_for, color='b', linestyle=':', linewidth=2)
        plt.grid()
        plt.legend(loc='best')
        fig.savefig("Images/four_peaks_fitness_varying_state.png")

    if gen_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Time efficiency against the n-dimensional state space with T=15%")
        ax.set_xlabel('n-dimensional state space')
        ax.set_ylabel('time efficiency at the best state')
        plt.plot(state_length, performance_time[:, 0], label='Simulated Annealing')
        plt.plot(state_length, performance_time[:, 1], label='Randomized Hill Climb')
        plt.plot(state_length, performance_time[:, 2], label='Genetic Algorithm')
        plt.plot(state_length, performance_time[:, 3], label='MIMIC')
        plt.grid()
        plt.legend(loc='best')
        fig.savefig("Images/four_peaks_varying_state_time.png")

    print("fitness review based on the input size completed")
    
 ### Additional work: review of the best tuned alogrithms with different T threshold value    
    
def four_peaks_T_threshold_tuning(best_schedule,best_restarts,best_mutation_prob,best_pop_size,t_pct_threshold=0.15, state_length=200):

    t_pct_thresholds=[0.15,0.2,0.25,0.30]
    
    for i, t_pct_threshold in enumerate(t_pct_thresholds):

        fitness = mlrose_hiive.FourPeaks(t_pct=t_pct_threshold)
        problem = mlrose_hiive.DiscreteOpt(length=state_length, fitness_fn=fitness, maximize=True, max_val=2)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=state_length)
        if best_schedule ==2:
            best_schedule = mlrose_hiive.ArithDecay()
        elif best_schedule ==1:
            best_schedule =mlrose_hiive.GeomDecay()
        else:
            best_schedule = mlrose_hiive.ExpDecay()  

        _, _, fitness_curv_sa = mlrose_hiive.simulated_annealing(problem, schedule=best_schedule, max_attempts= 20, max_iters=10000, init_state=init_state, curve=True)
        _, _, fitness_curv_rhc = mlrose_hiive.random_hill_climb(problem,restarts=best_restarts, max_attempts= 20, max_iters=10000, init_state= init_state, curve=True)
        _, _, fitness_curv_ga = mlrose_hiive.genetic_alg(problem,mutation_prob=best_mutation_prob, max_attempts= 20, max_iters=10000, curve=True)
        _, _, fitness_curv_mimic = mlrose_hiive.mimic(problem, pop_size=best_pop_size, max_attempts= 20, max_iters=10000, curve=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Fitness Curve with varying T threshold | Four Peak Problem")
        ax.set_xlabel('Iterations')
        ax.set_ylabel('fitness curve')
        ax.plot(fitness_curv_sa[:, 0], label='SA T={}'.format(t_pct_threshold))
        ax.plot(fitness_curv_rhc[:, 0], label='RHC T={}'.format(t_pct_threshold))
        ax.plot(fitness_curv_ga[:, 0], label='GA T={}'.format(t_pct_threshold))
        ax.plot(fitness_curv_mimic[:, 0], label='MIMIC T={}'.format(t_pct_threshold))
        ax.grid()
        ax.legend(loc='best')
        fig.savefig("Images/four_peaks_iterations_T_{}.png".format(t_pct_threshold))

    print("fitness review based on the T threshold completed")
    

if __name__ == "__main__":

    #Run hyperparameter decay function for SA
    best_schedule=four_peaks_fitness_curve_hyperparameter_sa(t_pct_threshold=0.15, state_length=200)

    # Run hyperparameter restarts for RHC
    best_restarts=four_peaks_fitness_curve_hyperparameter_rhc(t_pct_threshold=0.15, state_length=200)

    # Run hyperparameter mutation_prob for GA
    best_mutation_prob=four_peaks_fitness_curve_hyperparameter_ga(t_pct_threshold=0.15, state_length=200)

    # Run hyperparameter pop_size for MIMIC
    best_pop_size=four_peaks_fitness_curve_hyperparameter_mimic(t_pct_threshold=0.15, state_length=200)
    
    # Run the fitness curve at evey iteration
    four_peaks_fitness_curve_iterations(best_schedule,best_restarts,best_mutation_prob,best_pop_size,t_pct_threshold=0.15, state_length=200)
    
    # Run state space size
    four_peaks_state_size(best_schedule,best_restarts,best_mutation_prob,best_pop_size,t_pct_threshold=0.15,state_size_tunned_for=200)

    # Run different t_pct threshold values
    four_peaks_T_threshold_tuning(best_schedule,best_restarts,best_mutation_prob,best_pop_size,t_pct_threshold=0.15, state_length=200)








