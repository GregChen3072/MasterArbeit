# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:26:04 2020

@author: johan
"""


import random
import numpy as np


class NextN:
    def __init__(self, n_size: int, n_type: str = "random", batch_size: int = 5, n_data_sets: int = None) -> None:
        super().__init__()
        self.n_size = n_size
        self.n_type = n_type
        self.n_current = None
        self.batch_size = batch_size
        self.n_data_sets = n_data_sets
        self.first_ping = True
        #print("n_type: "+str(n_type))

    def __n_batch(self):
        if self.n_current is None:
            self.n_current = self.batch_size
        else:
            self.n_current = self.n_current + self.batch_size

    def __n_random(self):
        if self.n_current is None:
            self.n_current = 0
        self.n_current = random.randint(self.n_current+1, self.n_size)

    def __n_single(self):
        self.n_current = self.n_size

    def __n_communist(self):
        if self.n_data_sets is None:
            raise Exception("The number of data sets is not given.")
        self.batch_size = int(self.n_size / self.n_data_sets)
        self.__n_batch()
        self.n_type = "batch"

    def __n_bear(self):
        if self.n_current is None:
            self.n_current = int(self.n_size / 2)
        else:
            half = int((self.n_size - self.n_current) / 2)
            if half == 0:
                self.n_current = self.n_current + 1
            else:
                self.n_current = self.n_current + half

    def __n_bull(self):
        if self.n_current is None:
            self.n_current = 1
        else:
            self.n_current = self.n_current * 2

    def __level(self):
        if self.n_current is None:
            raise Exception("Error: n_current is not set.")
        elif self.n_current > self.n_size:
            self.n_current = self.n_size
        elif self.n_current <= 0:
            self.n_current = 1

    def get_next_n(self):
        # print("start: get next n. TYPE: "+str(self.n_type))
        if self.n_current == self.n_size:
            raise Exception("Already finished.")
        if self.n_type == "batch":
            self.__n_batch()
        elif self.n_type == "random":
            self.__n_random()
        elif self.n_type == "single":
            self.__n_single()
        elif self.n_type == "communist":
            self.__n_communist()
        elif self.n_type == "bear":
            self.__n_bear()
        elif self.n_type == "bull":
            self.__n_bull()
        else:
            raise Exception("Error: No acceptable n_type.")
        self.__level()
        # print(self.n_current)
        return self.n_current

    def finished(self):
        # print("n_current: "+str(self.n_current))
        # print("n_size: "+str(self.n_size))
        if self.n_current is None:
            if self.first_ping:
                self.first_ping = False
                return False
            else:
                raise Exception("n_current is None")
        res = self.n_current == self.n_size
        # print("finished: "+str(res))
        return res


class NextDataSets:
    def __init__(self, data_sets: list, next_type: str, accuracy_batch_size: int = 1) -> None:
        super().__init__()
        self.next_type = next_type
        self.data_sets = data_sets
        self.n_current_index = None
        self.sorted_index = None
        self.current_sorted_index = None
        self.batch_size = accuracy_batch_size
        #print(self.next_type)
        
    def __random(self):
        if self.n_current_index is None:
            self.n_current_index = 0
        self.n_current_index = random.randint(self.n_current_index, (len(self.data_sets)-1))

    def __iterate(self):
        if self.n_current_index is None or self.n_current_index + 1 == len(self.data_sets):
            self.n_current_index = 0
        else:
            self.n_current_index = self.n_current_index + 1

    def __sort_index_patients(self):
        patients = [x.get_number_of_batches_of_patients(batch_size=self.batch_size) for x in self.data_sets]
        patients = np.array(patients)
        self.sorted_index = np.argsort(patients).tolist()
        #print(self.sorted_index)

    def __biggest(self):
        if self.sorted_index is None:
            self.__sort_index_patients()
            self.current_sorted_index = 0
            self.sorted_index.reverse()
            self.next_type = "iterate_sorted_list"
            self.n_current_index = self.sorted_index[self.current_sorted_index]
        else:
            raise Exception("Error: Sorted index should be empty.")

    def __smallest(self):
        if self.sorted_index is None:
            self.__sort_index_patients()
            self.current_sorted_index = 0
            self.next_type = "iterate_sorted_list"
            self.n_current_index = self.sorted_index[self.current_sorted_index]
        else:
            raise Exception("Error: Sorted index should be empty.")

    def __iterate_sorted(self):
        if self.current_sorted_index + 1 == len(self.sorted_index):
            self.current_sorted_index = 0
        else:
            self.current_sorted_index = self.current_sorted_index + 1
        self.n_current_index = self.sorted_index[self.current_sorted_index]

    def __sort_index_score(self, classifier):
        try:
            scores = [x.get_score(classifier) for x in self.data_sets] # TODO: 
            scores = np.array(scores)
            self.sorted_index = np.argsort(scores).tolist()
        except:
            self.sorted_index = [0]

    def __best_score(self, classifier):
        if self.n_current_index is None:
            self.n_current_index = 0
        self.__sort_index_score(classifier)
        self.n_current_index = self.sorted_index[-1]

    def __worst_score(self, classifier):
        if self.n_current_index is None:
            self.n_current_index = 0
        self.__sort_index_score(classifier)
        self.n_current_index = self.sorted_index[0]

    def get_next_index(self, classifier) -> int:
        # print("Next database type: "+str(self.next_type))
        if self.next_type == "iterate":
            self.__iterate()
        elif self.next_type == "random":
            self.__random()
        elif self.next_type == "biggest":
            self.__biggest()
        elif self.next_type == "smallest":
            self.__smallest()
        elif self.next_type == "best":
            self.__best_score(classifier)
        elif self.next_type == "worst":
            self.__worst_score(classifier)
        elif self.next_type == "iterate_sorted_list":
            self.__iterate_sorted()
        else:
            raise Exception("Error: Type not given.")
        if self.n_current_index is None:
            raise Exception
        # print(self.n_current_index)

        return self.n_current_index
