# Yulia Cher

import math
import sys

# Vocabulary size
V = 300000

#----------------------------
# Utility class
# contains general functions
#----------------------------
class Utility:
    def create_frequency_map_from_list(self, sample):
        map = {}
        for event in sample:
            if (event in map):
                map[event] = map[event] + 1
            else:
                map[event] = 1
        return map

#----------------------------
# FileManager class
# contains operations with files
#----------------------------
class FileManager:
 
    def write_output_to_file(self, filename, output_data):
        try:
            # create file
            f = open(filename, "x")
        except:
            # file exists - open and overwright content
            f = open(filename, "w")       
        for line in output_data:
            f.write(line + "\n")
        f.close() 

    def read_file(self, filename):
        s = []
        file_obj = open(filename, "r")  
        for line in file_obj:
            if line.startswith('<TRAIN') or line.startswith('<TEST') or line == '\n':
                continue    # skeep line with title
            s.extend(line.strip().split(' '))
        file_obj.close()
        return s

#----------------------------
# LidstoneModel class
# contains Lidstone Model functionality
#----------------------------
class LidstoneModel:
    def __init__(self, events, input_word, vocabulary_size):
        self.total_events = events
        self.input_word = input_word
        self.vocabulary_size = vocabulary_size
        self.best_lamb = 0

        self.total_events_number = 0
        self.freq_map_total_events = {}
        self.diff_events_number = 0

        self.training_set = []
        self.validation_set = []
        self.training_set_size = 0
        self.validation_set_size = 0

        self.freq_map_training = {}
        self.freq_map_validation = {}
        self.diff_events_number_in_training_set = 0
        self.unseen_word_freq_in_training_set = 0
        self.input_word_freq_in_training_set = 0   

        self.best_lambda = 0    

    #----------------------------
    # init_data() function
    # calculate the needed parameters
    # create training and validation sets
    #----------------------------
    def init_data(self):
        self.total_events_number = len(self.total_events)
        self.freq_map_total_events = Utility().create_frequency_map_from_list(self.total_events)
        self.diff_events_number = len(self.freq_map_total_events)
        self.create_training_and_validation_set()

    #----------------------------
    # get_uniform_distribution_probability() function
    # all events in the language have an equal probability to occur
    #----------------------------
    def get_uniform_distribution_probability(self):
        return 1 / self.vocabulary_size  
    
    #----------------------------
    # create_training_and_validation_set() function
    # devide development set to train and validation
    # as 90% and 10% respectively
    #----------------------------
    def create_training_and_validation_set(self):
        devide_index = round(0.9 * self.total_events_number)
        self.training_set = self.total_events[:devide_index]
        self.validation_set = self.total_events[devide_index:]
        self.training_set_size = len(self.training_set)
        self.validation_set_size = len(self.validation_set)
        util = Utility()
        self.freq_map_training = util.create_frequency_map_from_list(self.training_set)
        self.freq_map_validation = util.create_frequency_map_from_list(self.validation_set) 
        self.diff_events_number_in_training_set = len(self.freq_map_training)
        self.input_word_freq_in_training_set = self.freq_map_training[self.input_word]

    #----------------------------
    # calculate_MLE() function
    # calculate MLE according to event frequency and sample size
    #----------------------------
    def calculate_MLE(self, event_freq, sample_size):
        p_mle = -1
        if sample_size >= 0:
           p_mle = event_freq/sample_size
        return p_mle 

    #----------------------------
    # get_MLE_input_word_training_set() function
    # calculate MLE for input word in training set
    # with no smoothing
    #----------------------------   
    def get_MLE_input_word_training_set(self):
        return self.calculate_MLE(self.input_word_freq_in_training_set, self.training_set_size)

    #----------------------------
    # get_MLE_unseen_word_training_set() function
    # calculate MLE for unseen word in training set
    # with no smoothing
    #---------------------------- 
    def get_MLE_unseen_word_training_set(self):
        return self.calculate_MLE(self.unseen_word_freq_in_training_set, self.training_set_size)

    #----------------------------
    # get_probability() function
    # calculate event probability according to model
    #---------------------------- 
    def get_probability(self, word_freq, sample_size, lamb):
        numerator = word_freq + lamb
        denominator = sample_size + lamb * self.vocabulary_size
        p = numerator/denominator
        return p
    
    #----------------------------
    # get_probability_train_input_word() function
    # calculate probability of input word according to lambda
    #---------------------------- 
    def get_probability_train_input_word(self, lamb):
        return self.get_probability(self.input_word_freq_in_training_set, self.training_set_size, lamb)
    
    #----------------------------
    # get_probability_train_unseen_word() function
    # calculate probability of unseen word according to lambda
    #----------------------------   
    def get_probability_train_unseen_word(self, lamb):
        return self.get_probability(self.unseen_word_freq_in_training_set, self.training_set_size, lamb)
    
    #----------------------------
    # calc_perplexity() function
    # calculate perplexity for specific lambda
    # probability is calculated according training set
    # perplexity is calculated for sample  
    # The function runs on the validation set 
    # but uses event probability that was calculated on the training set (model training)
    #----------------------------  
    def calc_perplexity(self, sample, lamb):
        perplexity = 0
        log_sum = 0

        for event in list(sample):
            event_freq_training = self.freq_map_training.get(event, 0)
            p_event_train = self.get_probability(event_freq_training, self.training_set_size, lamb)
            if p_event_train > 0:
                log = math.log(p_event_train, 2)
                log_sum += log

        power = (-1/len(sample)) * log_sum
        perplexity = pow(2, power)
        return perplexity
    
    #----------------------------
    # calc_perplexity_validation_set() function
    # calculate perplexity values for different lambdas 
    #----------------------------  
    def calc_perplexity_validation_set(self, lamb_array):
        perplexity_array = []
        for lamb in lamb_array:
            perplexity = self.calc_perplexity(self.validation_set, lamb)
            perplexity_array.append(perplexity)
        return perplexity_array
    
    #----------------------------
    # calc_best_lambda() function
    # choose the best lambda from range 0-2 
    #----------------------------  
    def calc_best_lambda(self):
        # choose best lambda value and min preplexity
        best_lamb = 0 
        min_preplexity = self.calc_perplexity(self.validation_set, 0.01)
        
        for lamb in range(2, 200):
            lamb = lamb / 100
            perplexity = self.calc_perplexity(self.validation_set, lamb)
            if (perplexity < min_preplexity):
                min_preplexity = perplexity
                best_lamb = lamb

        self.best_lambda = best_lamb
        return (self.best_lambda, min_preplexity)
    
    #SW
    def calc_best_lambda1(self):
        """
        Returns: (best_lambda, best_dev_pp, sweep)
        where sweep is a list of (lambda_value, dev_perplexity).
        """
        sweep = []
        best_lambda = None
        best_pp = float('inf')

        # Dense sweep: 0.01 .. 1.99 (step 0.01)
        for i in range(1, 200):
            lamb = i / 100.0
            pp_dev = self.calc_perplexity(self.validation_set, lamb)
            sweep.append((lamb, pp_dev))
            if pp_dev < best_pp:
                best_pp = pp_dev
                best_lambda = lamb

        self.best_lambda = best_lambda
        return best_lambda, best_pp, sweep

    #----------------------------
    # check_total_probability_for_all_events_lid() - validation function
    # calculate total probability for all events in deveelopnment according to model
    # the result should be equal to 1
    #---------------------------- 
    def check_total_probability(self):
        # calculate propability for unseen words
        number_of_unseen_events = self.vocabulary_size - self.diff_events_number
        lamb = 0.01
        p_unseen_event = self.get_probability(self.unseen_word_freq_in_training_set, self.total_events_number, lamb)
        p_unseen_total = number_of_unseen_events * p_unseen_event
        
        # calculate propability for observed events in development set
        p_observed_total = 0
        for event_freq in self.freq_map_total_events.values():
            p_event = self.get_probability(event_freq, self.total_events_number, lamb)
            p_observed_total += p_event
        
        total_probability = p_unseen_total + p_observed_total
        print('Total probability lid : ' + str(total_probability))

#----------------------------
# HeldoutModel class
#----------------------------
class HeldoutModel:
    def __init__(self, events, input_word, vocabulary_size):
        self.total_events = events
        self.input_word = input_word
        self.vocabulary_size = vocabulary_size

        self.total_events_number = 0
        self.st_train = []
        self.sh_heldout = []
        self.st_train_size = 0
        self.sh_heldout_size = 0
        self.freq_map_st_train = {}
        self.freq_map_sh_heldout = {}

    #----------------------------
    # init_data() function
    # initialize class parameters and
    # devide development to train and held-out
    # as 50% and 50%
    #----------------------------
    def init_data(self):
        self.total_events_number = len(self.total_events)

        devide_index = round(self.total_events_number/2)
        self.st_train = self.total_events[:devide_index]
        self.sh_heldout = self.total_events[devide_index:]

        self.st_train_size = len(self.st_train)
        self.sh_heldout_size = len(self.sh_heldout)

        self.freq_map_st_train = Utility().create_frequency_map_from_list(self.st_train)
        self.freq_map_sh_heldout = Utility().create_frequency_map_from_list(self.sh_heldout)

    #----------------------------
    # compute_total_freq_ho() function
    # calculate total frequency in held-out 
    # for all events with frequency r in train
    #----------------------------
    def compute_total_freq_ho(self, freq):
        total_freq_ho = 0
        for key in self.freq_map_st_train:
            if self.freq_map_st_train[key] == freq:
                if (key in self.freq_map_sh_heldout):
                    freq_ho = self.freq_map_sh_heldout[key]
                    total_freq_ho += freq_ho
        return total_freq_ho

    #----------------------------
    # get_probability_ho() function
    # calculate number of different events with freq r in train
    # calculate their total frequency in held-out
    # calculate probability according to 
    # total frequency in held-out for r / (number of different events with frequency r in train * size of held-out set)
    #----------------------------
    def get_probability_ho(self, freq_r):
        # find all different events with freq r in train
        total_freq_ho = 0
        n_r = 0 #number of different events with freq r in train
        for key in self.freq_map_st_train:
            if self.freq_map_st_train[key] == freq_r:
                n_r += 1
                # compute total freq in heldout
                if (key in self.freq_map_sh_heldout):
                    freq_ho = self.freq_map_sh_heldout[key]
                    total_freq_ho += freq_ho

        # compute probability
        p_ho = total_freq_ho / (n_r * self.sh_heldout_size)
        return p_ho

    #----------------------------
    # get_probability_ho_input_word() function
    # get input word frequency from the train
    # calculate probability for input word according to model
    #----------------------------
    def get_probability_ho_input_word(self):
        freq = self.freq_map_st_train[self.input_word]
        return self.get_probability_ho(freq)

    #----------------------------
    # calc_total_freq_ho_unseen_word() function
    # calculate total frequency for unseen word in held-out
    # for each event in held-out check :
    # if it does not exist in train - add its ho frequency to total
    #----------------------------
    def calc_total_freq_ho_unseen_word(self):
        total_freq_ho = 0
        for event in self.freq_map_sh_heldout:
            if not event in self.freq_map_st_train:
                freq_ho = self.freq_map_sh_heldout[event]
                total_freq_ho += freq_ho
        return total_freq_ho
                
    #----------------------------
    # get_probability_ho_unseen_word() function
    # calculate total frequency for unseen word in held-out
    # calculate amout of unseen words in train according to
    # vocabulary size - amount of observed words in train
    # calculate probability according to model
    #----------------------------
    def get_probability_ho_unseen_word(self):        
        # get unseen words in train
        total_freq_ho = self.calc_total_freq_ho_unseen_word()       
        # calculate amout of unseen words
        observed_events_number_train = len(self.freq_map_st_train)
        n_0 = self.vocabulary_size - observed_events_number_train
        p_ho = total_freq_ho / (n_0 * self.sh_heldout_size)
        return p_ho

    #----------------------------
    # check_total_probability_for_all_events_ho() - validation function
    # check total probability
    # it should be equal to 1 
    #----------------------------
    def check_total_probability_for_all_events_ho(self):
        observed_events_number = len(self.freq_map_st_train)
        n_0 = self.vocabulary_size - observed_events_number
        p_0 = self.get_probability_ho_unseen_word()
        p_0_total = p_0*n_0
        p_observed_total = 0
        for event in self.freq_map_st_train.keys():
             freq = self.freq_map_st_train[event]
             p_event = self.get_probability_ho(freq)
             p_observed_total += p_event

        total = p_0_total + p_observed_total
        print('Total probability ho : ' + str(total))

    #----------------------------
    # calc_perplexity() function
    # probability is calculated according training set
    # perplexity is calculated for sample
    #----------------------------
    def calc_perplexity(self, sample):
        perplexity = 0
        log_sum = 0
        cache = {}
        for event in list(sample):
            if event in cache:
                p_ho = cache[event]
            else:
                event_freq_training = self.freq_map_st_train.get(event, 0)
                if (event_freq_training == 0):
                    p_ho = self.get_probability_ho_unseen_word()
                else:
                    freq = self.freq_map_st_train[event]
                    p_ho = self.get_probability_ho(freq)
                cache[event] = p_ho
            if p_ho > 0:
                log = math.log(p_ho, 2)
                log_sum += log

        power = (-1/len(sample)) * log_sum
        perplexity = pow(2, power)
        return perplexity
    
    # === SW - BEGIN ADD: HeldoutModel counts-of-counts ===
    def counts_of_counts_table(self):
        """
        Returns a list of rows:
        (r, n_r, total_c_heldout_for_r, p_HO_r)
        where:
          - r is the frequency in ST-train,
          - n_r is the number of types with train frequency r,
          - total_c_heldout_for_r = sum_{w: c_train(w)=r} c_HO(w),
          - p_HO_r = total_c_heldout_for_r / (n_r * |HO|),
        and for r=0 we use:
          - n_0 = V - observed_types_in_train,
          - total_c_heldout_for_0 = sum of HO counts for tokens unseen in train.
        """
        from collections import defaultdict

        # Safety guards
        ho_size = self.sh_heldout_size if self.sh_heldout_size else 0
        train_map = self.freq_map_st_train or {}
        ho_map = self.freq_map_sh_heldout or {}

        # Build r -> list of types with that train frequency
        freq_to_types = defaultdict(list)
        for w, r in train_map.items():
            freq_to_types[r].append(w)

        # Establish range of r to report
        max_r = max(train_map.values(), default=0)

        rows = []

        # r = 0 (unseen in train)
        observed_types = len(train_map)
        n_0 = max(self.vocabulary_size - observed_types, 0)
        total_ho_unseen = self.calc_total_freq_ho_unseen_word()
        p0 = (total_ho_unseen / (n_0 * ho_size)) if (n_0 > 0 and ho_size > 0) else 0.0
        rows.append((0, n_0, total_ho_unseen, p0))

        # r >= 1
        for r in range(1, max_r + 1):
            types_r = freq_to_types.get(r, [])
            n_r = len(types_r)
            if n_r == 0:
                rows.append((r, 0, 0, 0.0))
                continue
            total_ho = sum(ho_map.get(w, 0) for w in types_r)
            p_r = (total_ho / (n_r * ho_size)) if ho_size > 0 else 0.0
            rows.append((r, n_r, total_ho, p_r))

        return rows
    # === END ADD: HeldoutModel counts-of-counts ===


#----------------------------
# ModelManager class
# contains logic
# loads input from files and creates output file
# creates and compares Lidstone and Held-out models
#----------------------------
class ModelManager:
    def __init__(self, dev_set_filename, test_set_filename, input_word, output_filename, vocabulary_size):
        self.dev_set_filename = dev_set_filename
        self.test_set_filename = test_set_filename
        self.output_filename = output_filename
        self.vocabulary_size = vocabulary_size
        self.input_word = input_word
        self.output = ["#Student\tYulia Cher\t317538775"]
        self.fm = FileManager()
        self.lid_model = None
        self.heldout_model = None

    # === SW - BEGIN ADD: ModelManager writers ===
    def _write_heldout_counts_csv(self, rows, path="stats/heldout_counts_of_counts.csv"):
        header = ["r", "n_r", "sum_c_heldout", "p_HO_r"]
        out_rows = []
        for r, n_r, total_ho, p_r in rows:
            if r > 9:
                continue
            out_rows.append([r, n_r, total_ho, f"{p_r:.8f}"])
        self._write_csv(path, header, out_rows)
    # === END ADD: ModelManager writers ===


    #----------------------------
    # run() function
    # invokes needed functionality according to the logic flow
    #----------------------------
    def run(self):
        events = self.fm.read_file(self.dev_set_filename)
        
        self.lid_model = LidstoneModel(events, self.input_word, self.vocabulary_size)
        self.lid_model.init_data()
        #------SW
        self._dataset_stats = {}

        # token counts
        train_tokens = getattr(self.lid_model, "training_set_size", None)
        valid_tokens = getattr(self.lid_model, "validation_set_size", None)
        dev_tokens   = getattr(self.lid_model, "total_events_number", None)

        # type counts (unique tokens)
        types_train = len(getattr(self.lid_model, "freq_map_training", {}))
        types_valid = len(getattr(self.lid_model, "freq_map_validation", {}))
        types_dev   = len(getattr(self.lid_model, "freq_map_total_events", {}))

        self._dataset_stats.update({
            "N_train_tokens": train_tokens or 0,
            "N_val_tokens":   valid_tokens or 0,
            "N_dev_tokens":   dev_tokens   or 0,
            "types_train":    types_train,
            "types_val":      types_valid,
            "types_dev":      types_dev,
            "V_assumed":      self.vocabulary_size,   # your |V|=300000
        })
        #------
        # init output data
        self.output.append("#Output1\t" + self.dev_set_filename)
        self.output.append("#Output2\t" + self.test_set_filename)
        self.output.append("#Output3\t" + self.input_word)
        self.output.append("#Output4\t" + self.output_filename)
        self.output.append("#Output5\t" + str(self.vocabulary_size))
        self.output.append("#Output6\t" + str(self.lid_model.get_uniform_distribution_probability()))
        self.output.append("#Output7\t" + str(self.lid_model.total_events_number))        
        self.output.append("#Output8\t" + str(self.lid_model.validation_set_size))
        self.output.append("#Output9\t" + str(self.lid_model.training_set_size))
        self.output.append("#Output10\t" + str(self.lid_model.diff_events_number_in_training_set))
        self.output.append("#Output11\t" + str(self.lid_model.input_word_freq_in_training_set))

        p_input_word = self.lid_model.get_MLE_input_word_training_set()
        p_unseen_word = self.lid_model.get_MLE_unseen_word_training_set()
        self.output.append("#Output12\t" + str(p_input_word))
        self.output.append("#Output13\t" + str(p_unseen_word))

        lamb = 0.10
        p_input_word = self.lid_model.get_probability_train_input_word(lamb)
        p_unseen_word = self.lid_model.get_probability_train_unseen_word(lamb)
        self.output.append("#Output14\t" + str(p_input_word))
        self.output.append("#Output15\t" + str(p_unseen_word))

        lamb_values = [0.01, 0.10, 1.00]
        perplexity = self.lid_model.calc_perplexity_validation_set(lamb_values)   
        self.output.append("#Output16\t" + str(perplexity[0]))
        self.output.append("#Output17\t" + str(perplexity[1]))
        self.output.append("#Output18\t" + str(perplexity[2]))

        best_result = self.lid_model.calc_best_lambda()
        best_lamb = best_result[0]
        min_preplexity = best_result[1]
        self.output.append("#Output19\t" + str(best_lamb))
        self.output.append("#Output20\t" + str(min_preplexity))

        self.heldout_model = HeldoutModel(events, self.input_word, self.vocabulary_size)
        self.heldout_model.init_data()

        # ===SW - BEGIN ADD: produce held-out counts-of-counts artifacts ===
        rows_counts = self.heldout_model.counts_of_counts_table()
        self._write_heldout_counts_csv(rows_counts)
        # === END ADD ===

        
        self.output.append("#Output21\t" + str(self.heldout_model.st_train_size))
        self.output.append("#Output22\t" + str(self.heldout_model.sh_heldout_size))

        p_ho_input_word = self.heldout_model.get_probability_ho_input_word()
        self.output.append("#Output23\t" + str(p_ho_input_word))

        p_ho_unseen_word = self.heldout_model.get_probability_ho_unseen_word()
        self.output.append("#Output24\t" + str(p_ho_unseen_word))

        self.evaluate_models()
        self.create_table()
        self.fm.write_output_to_file(self.output_filename, self.output)
        
        #---- for tests
        self.print_output_file()
        # self.lid_model.check_total_probability()
        # self.heldout_model.check_total_probability_for_all_events_ho() 

        #===================================
        #   SW
        #===================================
        # 1) pick best λ and get the *dev* sweep
        best_lamb, best_dev_pp, sweep = self.lid_model.calc_best_lambda1()

        # 2) ensure test set is loaded as a list of tokens (or however your calc_perplexity expects it)
        test_tokens = self.fm.read_file(self.test_set_filename)  # adapt to your file manager
        # (If you already have test cached elsewhere, reuse it.)
        test_tokens_list = test_tokens  # rename for clarity

        # types on test
        from collections import Counter
        freq_test = Counter(test_tokens_list)
        types_test = len(freq_test)

        # OOV relative to TRAIN vocabulary
        train_vocab = set(getattr(self.lid_model, "freq_map_training", {}).keys())
        oov_types = sum(1 for t in freq_test if t not in train_vocab)
        oov_tokens = sum(1 for t in test_tokens_list if t not in train_vocab)
        oov_rate = (oov_tokens / len(test_tokens_list)) if test_tokens_list else 0.0
        oov_type_rate = (oov_types / types_test) if types_test else 0.0

        # write one-row CSV
        rows = [[
            self._dataset_stats["N_train_tokens"],
            self._dataset_stats["N_val_tokens"],
            self._dataset_stats["N_dev_tokens"],
            len(test_tokens_list),
            self._dataset_stats["types_train"],
            self._dataset_stats["types_val"],
            self._dataset_stats["types_dev"],
            types_test,
            oov_types,
            f"{oov_rate:.6f}",
            f"{oov_type_rate:.6f}",
            self._dataset_stats["V_assumed"],
        ]]
        self._write_csv(
            "stats/dataset_stats.csv",
            ["N_train_tokens","N_val_tokens","N_dev_tokens","N_test_tokens",
            "types_train","types_val","types_dev","types_test",
            "OOV_types_test","OOV_token_rate_test","OOV_type_rate_test","V_assumed"],
            rows
        )
        
        # 3) compute test perplexity for every λ in the sweep
        rows = []
        for lamb, pp_dev in sweep:
            pp_test = self.lid_model.calc_perplexity(test_tokens, lamb)
            rows.append([f"{lamb:.2f}", f"{pp_dev:.6f}", f"{pp_test:.6f}"])

        # 4) write the CSV for plotting
        self._write_csv(
                "stats/perplexity_lidstone_sweep.csv",
                ["lambda", "perplexity_dev", "perplexity_test"],
                rows)
        
        center = best_lamb
        fine = []
        for j in range(-20, 21):  # +/- 0.20 around best, step 0.005
            lamb = max(1e-6, center + j * 0.005)
            pp_dev = self.lid_model.calc_perplexity(self.lid_model.validation_set, lamb)
            pp_test = self.lid_model.calc_perplexity(test_tokens, lamb)
            fine.append([f"{lamb:.3f}", f"{pp_dev:.6f}", f"{pp_test:.6f}"])

        self._write_csv(
            "stats/perplexity_lidstone_sweep_fine.csv",
            ["lambda", "perplexity_dev", "perplexity_test"],
            fine
        )
        self._export_input_word_summary()
        self._export_lidstone_sensitivity()

    #----------------------------
    # _export_lidstone_sensitivity() function
    # 
    #----------------------------
    def _export_lidstone_sensitivity(self, V_list=(100_000, 300_000, 500_000)):
        """
        Create 'stats/perplexity_lidstone_sensitivity.csv' with columns:
        V, lambda, PP_dev, PP_test
        Uses the existing LidstoneModel (same train/validation split), but
        temporarily varies its vocabulary size for the sweep.
        """
        import os, csv

        # make sure test tokens are loaded (reuse if you already have them)
        test_tokens = self.fm.read_file(self.test_set_filename)

        rows = []
        # remember original V so we can restore it
        orig_V = getattr(self.lid_model, "vocabulary_size", None)

        # dense lambda grid: 0.01 .. 1.99
        lambdas = [i / 100.0 for i in range(1, 200)]

        for V in V_list:
            # temporarily change |V| on the same model/split
            self.lid_model.vocabulary_size = int(V)

            for lam in lambdas:
                # dev = validation portion of development set
                pp_dev = self.lid_model.calc_perplexity(self.lid_model.validation_set, lam)
                # test curve for visualization (NOT for selecting λ)
                pp_test = self.lid_model.calc_perplexity(test_tokens, lam)
                rows.append([int(V), f"{lam:.2f}", f"{pp_dev:.6f}", f"{pp_test:.6f}"])

        # restore original V
        if orig_V is not None:
            self.lid_model.vocabulary_size = orig_V

        # write CSV
        self._write_csv(
            "stats/perplexity_lidstone_sensitivity.csv",
            ["V", "lambda", "PP_dev", "PP_test"],
            rows,
        )

    
    #----------------------------
    # _export_input_word_summary() function
    # 
    #----------------------------    
    def _export_input_word_summary(self):
        """
        Creates stats/input_word_summary.csv with:
        word, c_train, P_MLE, P_Lid(best), P_HO, is_seen
        """
        from math import isnan

        word = self.input_word.strip()
        # --- training stats (robust to missing keys) ---
        c_train = int(self.lid_model.freq_map_training.get(word, 0))
        N_train = int(self.lid_model.training_set_size)

        # --- MLE ---
        P_MLE = (c_train / N_train) if N_train > 0 else 0.0

        # --- Lidstone with best λ (assumes self.lid_model.best_lambda is set) ---
        lam = float(getattr(self.lid_model, "best_lambda", 0.0) or 0.0)
        V = int(self.vocabulary_size)
        P_Lid = (c_train + lam) / (N_train + lam * V) if (N_train + lam * V) > 0 else 0.0

        # --- Held-out ---
        # If your HeldoutModel exposes direct methods, use them; otherwise reconstruct.
        P_HO = None
        try:
            # preferred: dedicated function for the input word if you have it
            if c_train == 0 and hasattr(self.heldout_model, "get_probability_ho_unseen_word"):
                P_HO = float(self.heldout_model.get_probability_ho_unseen_word())
            elif c_train > 0 and hasattr(self.heldout_model, "get_probability_ho"):
                P_HO = float(self.heldout_model.get_probability_ho(c_train))
        except Exception:
            P_HO = None

        # Reconstruction path if not provided by model:
        if P_HO is None or isnan(P_HO):
            # Need: for r=c_train → N_r (n_r), t_r (sum_c_heldout), and |H| = Σ t_r
            n_r_map = getattr(self.heldout_model, "n_r_map", None)          # dict: r -> N_r
            t_r_map = getattr(self.heldout_model, "t_r_map", None)          # dict: r -> t_r
            if n_r_map is None or t_r_map is None:
                # fall back: compute from frequency maps if you have them
                # freq_map_st_train: counts in train; freq_map_heldout: counts in held-out
                fm_train = getattr(self.heldout_model, "freq_map_st_train", {})
                fm_hold  = getattr(self.heldout_model, "freq_map_st_heldout", {})
                from collections import Counter, defaultdict
                n_r_map = defaultdict(int)
                t_r_map = defaultdict(int)
                for x, cnt in fm_train.items():
                    n_r_map[cnt] += 1
                    t_r_map[cnt] += fm_hold.get(x, 0)
                # unseen (r=0): count types not in train; sum held-out counts of those
                # If |V| is assumed, unseen types = V - len(train_vocab)
                V = int(self.vocabulary_size)
                train_vocab = set(fm_train.keys())
                n_r_map[0] = max(0, V - len(train_vocab))
                t_r_map[0] = sum(c for x, c in fm_hold.items() if x not in train_vocab)

            H = sum(t_r_map.values()) if t_r_map else 0
            if H > 0 and n_r_map.get(c_train, 0) > 0:
                P_HO = (t_r_map[c_train] / n_r_map[c_train]) / H
            else:
                P_HO = 0.0

        # --- write CSV ---
        rows = [[
            word,
            c_train,
            f"{P_MLE:.12f}",
            f"{P_Lid:.12f}",
            f"{P_HO:.12f}",
            bool(c_train > 0),
        ]]
        self._write_csv(
            "stats/input_word_summary.csv",
            ["word", "c_train", "P_MLE", "P_Lid(best)", "P_HO", "is_seen"],
            rows,
        )

    #----------------------------
    # evaluate_models() function
    # compares Lidstone and Held-out models
    #----------------------------
    def evaluate_models(self):
        s = self.fm.read_file(self.test_set_filename)
        total_events = len(s)
        self.output.append("#Output25\t" + str(total_events))
        perp_lid = self.lid_model.calc_perplexity(s, self.lid_model.best_lambda)
        self.output.append("#Output26\t" + str(perp_lid))
        #The perplexity of the test set according to your held-out model
        perp_ho = self.heldout_model.calc_perplexity(s)
        self.output.append("#Output27\t" + str(perp_ho))
        better_model = 'H'
        if perp_lid <perp_ho :
            better_model = 'L'
        self.output.append("#Output28\t" + str(better_model))

    #----------------------------
    # create_table() function
    # calculates needed values and formats them
    #----------------------------
    # def create_table(self):
    #     table_output = ""
    #     for r in range(0,10):
    #         p_lid = self.lid_model.get_probability(r, self.lid_model.training_set_size, self.lid_model.best_lambda)
    #         f_lambda = format(p_lid * self.lid_model.training_set_size, '.5f')
    #         p_ho = 0
    #         n_r_t = 0
    #         t_r = 0
    #         if r == 0:
    #             p_ho = self.heldout_model.get_probability_ho_unseen_word()
    #             n_r_t = self.vocabulary_size - len(self.heldout_model.freq_map_st_train) 
    #             t_r = self.heldout_model.calc_total_freq_ho_unseen_word()
    #         else:
    #             p_ho = self.heldout_model.get_probability_ho(r)
    #             for event in self.heldout_model.freq_map_st_train:
    #                 if self.heldout_model.freq_map_st_train.get(event, 0) == r:
    #                     n_r_t += 1
    #             t_r = self.heldout_model.compute_total_freq_ho(r)
    #         f_h = format(p_ho * self.heldout_model.st_train_size, '.5f')
    #         table_output += str(r)  + "\t" + f_lambda + "\t" + f_h + "\t" +  str(n_r_t) + "\t" +  str(t_r) + "\n"
    #     # SW
    #     self._export_heldout_f_table(table_output)
    #     self.output.append("#Output29\n" + table_output) 
    def create_table(self):
        table_output = ""
        rows_output29 = []  # <-- collect numeric rows for CSV

        for r in range(0, 10):
            # Lidstone expected count for frequency class r
            p_lid = self.lid_model.get_probability(
                r, self.lid_model.training_set_size, self.lid_model.best_lambda
            )
            f_lambda_val = p_lid * self.lid_model.training_set_size  # numeric for CSV
            f_lambda_str = format(f_lambda_val, '.5f')               # string for Output29

            # Held-out stats
            if r == 0:
                p_ho = self.heldout_model.get_probability_ho_unseen_word()
                n_r_t = self.vocabulary_size - len(self.heldout_model.freq_map_st_train)
                t_r = self.heldout_model.calc_total_freq_ho_unseen_word()
            else:
                p_ho = self.heldout_model.get_probability_ho(r)
                # count types with training freq = r
                n_r_t = sum(1 for ev, c in self.heldout_model.freq_map_st_train.items() if c == r)
                t_r = self.heldout_model.compute_total_freq_ho(r)

            f_h_val = p_ho * self.heldout_model.st_train_size        # numeric for CSV
            f_h_str = format(f_h_val, '.5f')                         # string for Output29

            # keep your original tab-separated Output29 row
            table_output += f"{r}\t{f_lambda_str}\t{f_h_str}\t{n_r_t}\t{t_r}\n"

            # collect numeric row for CSV export
            rows_output29.append([r, f_lambda_val, f_h_val, n_r_t, t_r])

        # write CSV for Fig3b (expects list of 5-tuples)
        self._export_heldout_f_table(rows_output29)

        # keep original output block
        self.output.append("#Output29\n" + table_output)
        

    #----------------------------
    # _export_heldout_f_table
    # 
    #----------------------------
    def _export_heldout_f_table(self, rows_output29):
        """
        rows_output29: iterable of rows like
        [r, f_lambda, f_H, N_r_T, t_r]
        Writes stats/heldout_f_table.csv with columns:
        r, f_MLE, f_lambda, f_H, N_r_T, t_r
        """
        import os, csv
        os.makedirs("stats", exist_ok=True)
        with open("stats/heldout_f_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["r", "f_MLE", "f_lambda", "f_H", "N_r_T", "t_r"])
            for r, f_lambda, f_H, N_r_T, t_r in rows_output29:
                w.writerow([r, float(r), f_lambda, f_H, N_r_T, t_r])


    #----------------------------
    # print_output_file() - test function
    # prints the content of output file
    #----------------------------
    def print_output_file(self):
        out_file = open(self.output_filename, "r")
        text = out_file.read()
        out_file.close()
        print(text)   

    #SW
    def _write_csv(self, path, header, rows):
        import os, csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)



#----------------------------
# main() - function
# defines default argument values
# gets arguments from command line
# invokes ModelManager finctionality
#----------------------------  
def main():
    arg_development_file = "develop.txt"
    arg_test_file = "test.txt"
    arg_input_word = "honduras"
    #arg_input_word = "be"
    arg_output_file = "output.txt"
    
    # get command line arguments
    n = len(sys.argv)
    if n == 5:
        if n > 1 :
            arg_development_file = sys.argv[1]
        if n > 2 :
            arg_test_file = sys.argv[2]
        if n > 3 :
            arg_input_word = sys.argv[3]
        if n > 4 :
            arg_output_file = sys.argv[4]


    model_manger = ModelManager(arg_development_file, arg_test_file, arg_input_word, arg_output_file, V)
    model_manger.run()


if __name__=="__main__":
    main()

