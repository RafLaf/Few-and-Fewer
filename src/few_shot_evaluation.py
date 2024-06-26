import random
import json
import torch
import math
import numpy as np
import os 

def get_repository_path():
    import os
    return '/'+os.path.join(*os.path.abspath(__file__).split('/')[:-1])

def testFewShot(features, datasets = None, write_file=False):
    results = torch.zeros(len(features), 2)
    for i in range(len(features)):
        accs = []
        feature = features[i]
        Generator = {'metadataset_omniglot':OmniglotGenerator, 'metadataset_imagenet':ImageNetGenerator}.get(datasets[i]['name'].replace('_train', '').replace('_test', '').replace('_validation', '') if datasets != None else datasets, EpisodicGenerator)
        generator = Generator(datasetName=None if datasets is None else datasets[i]["name"], num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        for run in range(args.few_shot_runs):
            shots = []
            queries = []
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
            shots, queries = generator.get_features_from_indices(feature, episode)
            accs.append(classifiers.evalFewShotRun(shots, queries))
        accs = 100 * torch.tensor(accs)
        low, up = confInterval(accs)
        results[i, 0] = torch.mean(accs).item()
        results[i, 1] = (up - low) / 2
    if write_file:
        result_file = torch.load('results_dic.pt')
        try:
            result_file[args.test_dataset][args.load_backbone ] =  results
        except:    
            result_file[args.test_dataset] = {args.load_backbone : results}

        torch.save(result_file , 'results_dic.pt')
    return results


class EpisodicGenerator():
    def __init__(self, datasetName=None, dataset_path=None, max_classes=50, num_elements_per_class=None):
        assert datasetName != None or num_elements_per_class!=None, "datasetName and num_elements_per_class can't both be None"
        
        all_datasets = {}
        self.dataset_path = dataset_path
        if self.dataset_path != '' and self.dataset_path != None and 'omniglot' in datasetName:
            json_path = os.path.join(self.dataset_path, 'datasets_subdomain.json')

            if os.path.exists(json_path):
                f = open(json_path)    
                all_datasets = json.loads(f.read())
                f.close()

        self.datasetName = datasetName
        self.dataset = None
        if datasetName != None and datasetName in all_datasets.keys():
            self.dataset = all_datasets[datasetName]
        if num_elements_per_class == None and self.dataset!=None:
            self.num_elements_per_class = self.dataset["num_elements_per_class"]
        else:
            self.num_elements_per_class = num_elements_per_class
        if self.num_elements_per_class != None:
            self.max_classes = min(len(self.num_elements_per_class), 50)
                
    def  select_classes(self, ways):
        # number of ways for this episode
        n_ways = ways if ways!=0 else random.randint(5, self.max_classes)

        # get n_ways classes randomly
        choices = torch.randperm(len(self.num_elements_per_class))[:n_ways]
        return choices 
    
    def get_query_size(self, choice_classes, n_queries):
        if n_queries == 0:
            min_queries = n_queries if n_queries != 0 else 10
            query_size = min([int(0.5*self.num_elements_per_class[c]) for c in choice_classes]) 
            query_size = min(min_queries, query_size)
        else:
            query_size = n_queries
        return query_size

    def get_support_size(self, choice_classes, query_size, n_shots):
        # sample beta uniformly from (0,1]
        if n_shots == 0:
            beta = 0.
            while beta == 0.:
                beta = torch.rand(1).item()
            support_size = sum([math.ceil(beta*min(100, self.num_elements_per_class[c]-query_size)) for c in choice_classes])
            support_size = min(500, support_size)
        else:
            support_size = len(choice_classes)*n_shots
        return support_size
    
    def get_number_of_shots(self, choice_classes, support_size, query_size, n_shots):
        if n_shots == 0: 
            n_ways = len(choice_classes)
            alphas = torch.Tensor(np.random.rand(n_ways)*(np.log(2)-np.log(0.5))+np.log(0.5)) # sample uniformly between log(0.5) and log(2)
            proportions = torch.exp(alphas)*torch.cat([torch.Tensor([self.num_elements_per_class[c]]) for c in choice_classes])
            proportions /= proportions.sum() # make sum to 1
            n_shots_per_class = ((proportions*(support_size-n_ways)).int()+1).tolist()
            n_shots_per_class = [min(n_shots_per_class[i], self.num_elements_per_class[c]-query_size) for i,c in enumerate(choice_classes)]
        else:
            n_shots_per_class = [n_shots]*len(choice_classes)
        return n_shots_per_class

    def get_number_of_queries(self, choice_classes, query_size, unbalanced_queries, max_queries,n_shots_per_class ):
        if max_queries:
            n_queries_per_class  = [min([self.num_elements_per_class[c]-n_shots_per_class[i] for i,c in enumerate(choice_classes)])]  *len(choice_classes)
        elif unbalanced_queries:
            alpha = np.full(len(choice_classes), 2)
            prob_dist = np.random.dirichlet(alpha)
            while prob_dist.min()*query_size*len(choice_classes)<1: # if there is a class with less than one query resample
                prob_dist = np.random.dirichlet(alpha)
            n_queries_per_class = self.convert_prob_to_samples(prob=prob_dist, q_shot=query_size*len(choice_classes))
        else:
            n_queries_per_class = [query_size]*len(choice_classes)
        return n_queries_per_class

    def sample_indices(self, num_elements_per_chosen_classes, n_shots_per_class, n_queries_per_class, max_queries = False):
        shots_idx = []
        queries_idx = []
        for k, q, elements_per_class in zip(n_shots_per_class, n_queries_per_class, num_elements_per_chosen_classes):
            choices = torch.randperm(elements_per_class)
            shots_idx.append(choices[:k].tolist())
            queries_idx.append(choices[k:k+q].tolist())
        return shots_idx, queries_idx

    def sample_episode(self, ways=0, n_shots=0, n_queries=0, unbalanced_queries=False, verbose=False, max_queries = False):
        """
        Sample an episode
        """
        # get n_ways classes randomly
        choice_classes = self.select_classes(ways=ways)
        
        query_size = self.get_query_size(choice_classes, n_queries)
        support_size = self.get_support_size(choice_classes, query_size, n_shots)

        n_shots_per_class = self.get_number_of_shots(choice_classes, support_size, query_size, n_shots)
        n_queries_per_class = self.get_number_of_queries(choice_classes, query_size, unbalanced_queries, max_queries,n_shots_per_class )
        shots_idx, queries_idx = self.sample_indices([self.num_elements_per_class[c] for c in choice_classes], n_shots_per_class, n_queries_per_class, max_queries = max_queries)

        if verbose:
            print(f'chosen class: {choice_classes}')
            print(f'n_ways={len(choice_classes)}, q={query_size}, S={support_size}, n_shots_per_class={n_shots_per_class}')
            print(f'queries per class:{n_queries_per_class}')
            print(f'shots_idx: {shots_idx}')
            print(f'queries_idx: {queries_idx}')

        return {'choice_classes':choice_classes, 'shots_idx':shots_idx, 'queries_idx':queries_idx}

    def get_features_from_indices(self, features, episode, validation=False):
        """
        Get features from a list of all features and from a dictonnary describing an episode
        """
        choice_classes, shots_idx, queries_idx = episode['choice_classes'], episode['shots_idx'], episode['queries_idx']
        if validation : 
            validation_idx = episode['validations_idx']
            val = []
        shots, queries = [], []
        for i, c in enumerate(choice_classes):
            shots.append(features[c]['features'][shots_idx[i]])
            queries.append(features[c]['features'][queries_idx[i]])
            if validation : 
                val.append(features[c]['features'][validation_idx[i]])
        if validation:
            return shots, queries, val
        else:
            return shots, queries

    def convert_prob_to_samples(self, prob, q_shot):
        """
        convert class probabilities to numbers of samples per class
        reused : https://github.com/oveilleux/Realistic_Transductive_Few_Shot
        Arguments:
            - prob: probabilities of each class
            - q_shot: total number of queries for all classes combined
        """
        prob = prob * q_shot
        if sum(np.round(prob)) > q_shot:
            while sum(np.round(prob)) != q_shot:
                idx = 0
                for j in range(len(prob)):
                    frac, whole = math.modf(prob[j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[idx] = np.floor(prob[idx])
            prob = np.round(prob)
        elif sum(np.round(prob)) < q_shot:
            while sum(np.round(prob)) != q_shot:
                idx = 0
                for j in range(len(prob)):
                    frac, whole = math.modf(prob[j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[idx] = np.ceil(prob[idx])
            prob = np.round(prob)
        else:
            prob = np.round(prob)
        return prob.astype(int)

class ImageNetGenerator(EpisodicGenerator):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Read ImageNet graph
        split = {'train':'TRAIN', 'test':'TEST', 'validation': 'VALID'}[self.datasetName.split('_')[-1]]
        with open(os.path.join(get_repository_path(), 'datasets', 'ilsvrc_2012_dataset_spec.json'), 'r') as file:
            self.graph = json.load(file)['split_subgraphs'][split]

        self.graph_map = {node['wn_id']:node['children_ids'] for node in self.graph}
        self.node_candidates = [node for node in self.graph_map.keys() if 5<=len(self.get_spanning_leaves(node))<=392]
        self.classIdx = self.dataset["classIdx"]
    def get_spanning_leaves(self, node):
        """
        Given a graph and a node return the list of all leaves spanning from the node 
        """
        if len(self.graph_map[node]) == 0:
            return [node]
        else:
            names = []
            for children in self.graph_map[node]:
                names += self.get_spanning_leaves(children)
            return names

    def select_classes(self, ways):
        """
        Different protocol for ImageNet
        """
        # Sample a node in the graph
        node = self.node_candidates[random.randint(0, len(self.node_candidates)-1)]

        leaves_candidates = self.get_spanning_leaves(node)
        # Sample a number of ways
        #n_ways = ways if ways!=0 else random.randint(5, max_classes)
        n_ways = ways if ways!=0 else min(len(leaves_candidates), 50)

        # get n_ways classes randomly from the subgraph if n_ways is fixed or if number of nodes higher than 50.
        choices_idx = torch.randperm(len(leaves_candidates))[:n_ways]
        choices_names = [leaves_candidates[idx] for idx in choices_idx]
        choices = torch.Tensor([self.classIdx[leaf] for leaf in choices_names]).int()
        return choices 
class OmniglotGenerator(EpisodicGenerator):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_classes(self, ways):
        superclass_id = str(torch.randint(self.dataset['num_superclasses'],(1,1)).reshape(-1).item())
        classes_ids = self.dataset['classes_per_superclass'][superclass_id]
        num_sampled_classes = ways if ways!=0 else torch.randint(5,min(len(classes_ids),50),(1,1)).reshape(-1)
       
        return torch.tensor(classes_ids)[torch.randperm(len(classes_ids))[:num_sampled_classes].tolist()]
class MetaAlbumsGenerator(EpisodicGenerator):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_datasets = {}
        if self.dataset_path != '' and self.dataset_path != None:
            json_path = os.path.join(self.dataset_path, 'datasets.json')
            if os.path.exists(json_path):
                f = open(json_path)    
                all_datasets = json.loads(f.read())
                f.close()
        if self.datasetName in ['metaalbum_Micro', 'metaalbum_Mini', 'metaalbum_Extended']:
            if self.datasetName == 'metaalbum_Extended':
                self.albums = ['BCT', 'BRD', 'CRS', 'FLW', 'PLK', 'PLT_VIL', 'RESISC', 'SPT', 'TEX']
            else:
                self.albums = ['BCT', 'BRD', 'CRS', 'FLW', 'MD_MIX', 'PLK', 'PLT_VIL', 'RESISC', 'SPT', 'TEX']
            self.metaalbum = True
            self.dataset = {}
            setting = self.datasetName.split('_')[-1]
            for album in self.albums:
                datasetName = f'metaalbum_{album}_{setting}'
                if datasetName in all_datasets.keys():
                    self.dataset[datasetName] = all_datasets[datasetName]
        else:
            self.metaalbum = False
        self.max_classes = 20
    def select_classes(self, ways, album=None):
        # Only one dataset, no need to sample from metaalbum
        if self.metaalbum:
            if album == None:
                album_id = random.randint(0, len(self.albums)-1) # sample an album
                album = self.albums[album_id]
            num_elements_per_class = self.dataset[f'metaalbum_{album}_{self.datasetName.split("_")[-1]}']['num_elements_per_class']
        else:
            num_elements_per_class = self.dataset['num_elements_per_class']
        # number of ways for this episode
        n_ways = ways if ways!=0 else random.randint(2, self.max_classes)

        # get n_ways classes randomly
        choices = torch.randperm(len(num_elements_per_class))[:n_ways]
        return choices, album 
 
    def get_query_size(self, choice_classes, n_queries):
        query_size = n_queries if n_queries != 0 else 16
        return query_size

    def get_number_of_shots(self, choice_classes, n_shots):
        n_shots = n_shots if n_shots!=0 else random.randint(1, 20)
        n_shots_per_class = [n_shots]*len(choice_classes)
        return n_shots_per_class

    def get_number_of_queries(self, choice_classes, query_size, unbalanced_queries):
        if unbalanced_queries:
            alpha = np.full(len(choice_classes), 2)
            prob_dist = np.random.dirichlet(alpha)
            while prob_dist.min()*query_size*len(choice_classes)<1: # if there is a class with less than one query resample
                prob_dist = np.random.dirichlet(alpha)
            n_queries_per_class = self.convert_prob_to_samples(prob=prob_dist, q_shot=query_size*len(choice_classes))
        else:
            n_queries_per_class = [query_size]*len(choice_classes)
        return n_queries_per_class

    def sample_episode(self, ways=0, n_shots=0, n_queries=0, unbalanced_queries=False, verbose=False, album=None):
        """
        Sample an episode
        """
        # get n_ways classes randomly
        choice_classes, album = self.select_classes(ways=ways, album=album)
        
        query_size = self.get_query_size(choice_classes, n_queries)

        n_shots_per_class = self.get_number_of_shots(choice_classes, n_shots)
        n_queries_per_class = self.get_number_of_queries(choice_classes, query_size, unbalanced_queries)

        if album == None:
            num_elements_per_class = self.num_elements_per_class
        else:
            num_elements_per_class = self.dataset[f'metaalbum_{album}_{self.datasetName.split("_")[-1]}']['num_elements_per_class']

        shots_idx, queries_idx = self.sample_indices([num_elements_per_class[c] for c in choice_classes], n_shots_per_class, n_queries_per_class)

        if verbose:
            print(f'chosen album: {album}')
            print(f'chosen class: {choice_classes}')
            print(f'n_ways={len(choice_classes)}, q={query_size}, S={support_size}, n_shots_per_class={n_shots_per_class}')
            print(f'queries per class:{n_queries_per_class}')
            print(f'shots_idx: {shots_idx}')
            print(f'queries_idx: {queries_idx}')

        return {'album':album, 'choice_classes':choice_classes, 'shots_idx':shots_idx, 'queries_idx':queries_idx}

        
    def get_features_from_indices(self, features, episode, validation=False):
        """
        Get features from a list of all features and from a dictonnary describing an episode
        """
        album, choice_classes, shots_idx, queries_idx = episode['album'], episode['choice_classes'], episode['shots_idx'], episode['queries_idx']
        if album != None:
            album_id = self.albums.index(album)
            features = features[start:end]
        if validation : 
            validation_idx = episode['validations_idx']
            val = []
        shots, queries = [], []
        for i, c in enumerate(choice_classes):
            shots.append(features[c]['features'][shots_idx[i]])
            queries.append(features[c]['features'][queries_idx[i]])
            if validation : 
                val.append(features[c]['features'][validation_idx[i]])
        if validation:
            return shots, queries, val
        else:
            return shots, queries
if __name__=='__main__':
    from args import args

    print('Test')
    print(args.dataset)

    if args.test_features != '':
        import classifiers
        from utils import confInterval
        feature = torch.load(args.test_features, map_location=args.device)
    results = testFewShot([feature], datasets = None, write_file=False)
    print(args.test_features)
    print("\t{:.3f} ±{:.3f}".format(results[0,0], results[0,1]) )
