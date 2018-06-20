import argparse
from time import strftime, gmtime
import time
import sys
from data.weighted_vectors import WeightedVectors
from genetic_programming.cca_helpers import *
from scipy.sparse import vstack

def main(args):

    print('Loading collection')
    Xtr,ytr,Xva,yva,Xte,yte = loadCollection(dataset=args.dataset, pos_cat=args.cat, fs=args.fs)
    trpos, vapos, tepos = ytr.sum(),yva.sum(),yte.sum()
    print('Positive documents in: Xtr={} Xva={} Xte={}'.format(trpos, vapos, tepos))
    if trpos <= 3 or vapos <= 3:
        print('avoiding optimization due to the lack sufficient positive examples; returing tfidf versions instead')
        data = TextCollectionLoader(dataset=args.dataset, vectorizer='tfidf', rep_mode='sparse',
                                    positive_cat=args.cat,feat_sel=args.fs)
        Xtr_weighted, ytr = data.get_train_set()
        Xva_weighted, yva = data.get_validation_set()
        Xte_weighted, yte = data.get_test_set()
        formula = 'tfidf'
        elapsed_time = 0

    else:
        t_init = time.time()
        print('Initializing terminals and operations')
        slope_t15 = find_best_slope_t15(Xtr, ytr, Xva, yva)
        slope_t16 = find_best_slope_t16(Xtr, ytr, Xva, yva)
        slope_t17 = find_best_slope_t17(Xtr, ytr, Xva, yva)

        operation_pool = get_operations()
        Xtr_terminals_pool = get_terminals(Xtr, slope_t15, slope_t16, slope_t17)
        Xva_terminals = get_terminals(Xva, slope_t15, slope_t16, slope_t17, asdict=True)


        initial_population_size = args.populationsize
        max_depth = args.initdepth
        max_populations = args.maxiter

        #np.random.seed(1)
        print('Init')
        population = ramped_half_and_half_method(initial_population_size, max_depth, operation_pool, Xtr_terminals_pool)

        epoch = 0
        while epoch < max_populations:
            print('Population {}'.format(epoch))

            # compute the fitness for each individual
            print('\tComputing fitness')
            fitness_population(population, Xva_terminals, ytr, yva, show=True)

            # create new population
            print('\tReproduction')
            new_population = reproduction(population, rate_r=0.05)

            print('\tCrossover')
            new_population.extend(crossover(population, rate_c=0.9))

            print('\tMutation')
            new_population.extend(mutate(population, operation_pool, Xtr_terminals_pool, rate_m=0.05))

            population = new_population
            epoch+=1

        best = fitness_population(population, Xva_terminals, ytr, yva, show=True)
        elapsed_time = time.time() - t_init

        print('Best individuals:')
        for p in population[:5]:
            print(p)
            print()

        Xte_terminals = get_terminals(Xte, slope_t15, slope_t16, slope_t17, asdict=True)

        Xtr_weighted = best()
        Xva_weighted = best(Xva_terminals)
        Xte_weighted = best(Xte_terminals)
        formula = str(best).replace('\n', ' ').replace('\t', '')

    vectorizer_name = 'GenCCA'
    run_params_dic = {'num_features': Xtr.shape[1],
                      'date': strftime("%d-%m-%Y", gmtime()),
                      'notes': formula,
                      'run': args.run,
                      'learn_tf': True,
                      'learn_idf': True,
                      'learn_norm': True,
                      'outmode': elapsed_time,
                      'iterations':args.maxiter}

    wv = WeightedVectors(vectorizer=vectorizer_name, from_dataset=args.dataset, from_category=args.cat,
                         trX=Xtr_weighted, trY=ytr,
                         vaX=Xva_weighted, vaY=yva,
                         teX=Xte_weighted, teY=yte,
                         run_params_dic=run_params_dic)
    wv.pickle(args.outdir, args.outname)
    print 'Weighted vectors saved at ' + args.outname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genetic Programming algorithm CCA',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset', help='Name of the dataset. Valid ones are: reuters21578, 20newsgroups, and ohsumed')
    parser.add_argument('cat', help='positive category', type=int)
    parser.add_argument('--fs', help='feature selection', type=float, default=0.1)
    parser.add_argument('--outdir', help='Output dir for learned vectors (default "../vectors").', type=str, default='../vectors')
    parser.add_argument('--run', help='Number of run, and seed used to replicate experiments (default 0).', type=int, default=0)
    parser.add_argument('--populationsize', help='Size of the population (default 200).', type=int, default=200)
    parser.add_argument('--initdepth', help='Max depth of the initial population (default 5).', type=int, default=5)
    parser.add_argument('--maxiter', help='Maximum number of iterations (default 30).', type=int, default=30)
    parser.add_argument('--maxdepth', help='Max depth allowed for a tree (default 15).', type=int, default=15)

    args = parser.parse_args()
    args.outname = 'GenCCA_' + args.dataset[:3] + '_C' + str(args.cat) + '_R' + str(args.run) + '.pickle'

    Tree.MAX_TREE_DEPTH=args.maxdepth



    if os.path.exists(os.path.join(args.outdir, args.outname)):
        print("Vector file {} already computed in dir {}. Skipping.".format(args.outname, args.outdir))
        sys.exit(1)

    np.random.seed(args.run)

    main(args)
