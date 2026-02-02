import numpy as np
import json
from fsm import *


def load_dataset(data_path, config_file):    
    with open(config_file, 'r') as f:
        dataset_config = json.load(f)
    dataset = np.load(data_path, allow_pickle=True)
    
    return dataset, dataset_config


class Activity():
    def __init__(self, name, data_cat, window_sec=5, fsm_list=None, simple_label=False):
        """
        Args:
            name (string): the name of this activity.
            data_cat (string): 'train' or 'test' dataset category.
            window_sec (float): window size in seconds (e.g., 2.5, 5).
            fsm_list (list): a list of FSM class instances.
            simple_label (bool): generate a single CE label if true, otherwise generate a list of CE labels corresponding to each timestamp.
        """
        self.name = name
        self.window_sec = window_sec
        self.fsm_list = fsm_list
        self.simple_label = simple_label

        if data_cat == 'train':
            self.data_path = './Multimodal/MultimodalDataset_audio-40000_1234_imu-50_1234.npz'
        elif data_cat == 'test':
            self.data_path = './Multimodal/MultimodalDataset_audio-40000_5_imu-50_5.npz'
        else:
            raise Exception("Unexpected dataset category - should choose from 'train' or 'test'.") 

        self.config_file = './Multimodal/dataset_config_w2.5.json'

        dataset, data_config = load_dataset(self.data_path, self.config_file)
        self.nClasses = data_config['nClasses']
        self.label_mapping = data_config['label_mapping']
        self.data, self.class_index_list = self.get_data(dataset)

        if self.fsm_list is not None:
            self.event_label_sequence = []
            self.state_input_sequence = []
            self.state_output_sequence = []
        self.action_sequence = []
        self.action_label_sequence = []
        self.sensor_data_indices = []  # Track indices into multimodal dataset
        self.data_source = data_cat  # 'train' or 'test'
        self.action_length = {}
        self.activity_length = {}
        self.time_window_elapsed = 0
        self._define_actions()

    def _define_actions(self):
        """
        Define self.action_length['act_x'] = (min_seconds, max_seconds)
            - 'act_x' has a random duration in seconds within the range (min_seconds, max_seconds)
            - Converted to windows using _seconds_to_windows() based on self.window_sec
        """
        raise NotImplementedError

    def _seconds_to_windows(self, seconds_tuple):
        """
        Convert (min_sec, max_sec) to (min_windows, max_windows).
        Validates that durations are exactly divisible by window_sec.

        Args:
            seconds_tuple: (min_seconds, max_seconds)

        Returns:
            (min_windows, max_windows)

        Raises:
            AssertionError: If durations are not divisible by window_sec
        """
        min_sec, max_sec = seconds_tuple

        assert min_sec % self.window_sec == 0, \
            f"Duration {min_sec}s not divisible by window_sec={self.window_sec}s"
        assert max_sec % self.window_sec == 0, \
            f"Duration {max_sec}s not divisible by window_sec={self.window_sec}s"

        return (int(min_sec // self.window_sec), int(max_sec // self.window_sec))

    def generate_activity(self):
        """
        Each activity is composed of atomic actions from the multimodal datset:
        {
            "brush_teeth": 0, 
            "click_mouse": 1, 
            "drink": 2, 
            "eat": 3, 
            "flush_toilet": 4, 
            "sit": 5, 
            "type": 6, 
            "walk": 7, 
            "wash": 8,
        }
        """
        raise NotImplementedError
    
    def generate_complex_label(self, states=True):
        """
        Return the current states, next states, and complex event label(s) sequences.
        For multi-label: event_label_sequence is a list of binary vectors (n_windows, 10).
        """
        assert self.fsm_list is not None
        if self.simple_label is True:
            # For multi-label, simple_label returns logical OR across all time windows
            # Result: single binary vector indicating which events occurred anywhere in the sequence
            all_labels = np.stack(self.event_label_sequence, axis=0)  # (n_windows, 10)
            return [np.max(all_labels, axis=0)]  # (10,) - OR across time
        if states is True:
            return self.event_label_sequence, self.state_input_sequence, self.state_output_sequence
        return self.event_label_sequence

    def get_generation_trace(self):
        """
        Return all information needed to reproduce this activity from indices.

        Returns:
            dict with keys:
                - sensor_data_indices: List of indices into multimodal dataset
                - data_source: 'train' or 'test'
                - action_sequence: List of action names
                - action_label_sequence: List of action label ids
        """
        return {
            'sensor_data_indices': self.sensor_data_indices.copy(),
            'data_source': self.data_source,
            'action_sequence': self.action_sequence.copy(),
            'action_label_sequence': self.action_label_sequence.copy(),
        }
    
    def _truncate_events(self, enforce_window_length):
        """
        Truncate all sequences to a fix length (enforce_window_length)
        """
        self.action_sequence = self.action_sequence[:enforce_window_length]
        self.action_label_sequence = self.action_label_sequence[:enforce_window_length]
        self.sensor_data_indices = self.sensor_data_indices[:enforce_window_length]
        if self.fsm_list is not None:
            self.event_label_sequence = self.event_label_sequence[:enforce_window_length]
            self.state_input_sequence = self.state_input_sequence[:enforce_window_length]
            self.state_output_sequence = self.state_output_sequence[:enforce_window_length]


    def _add_complex_label(self, action):
        """
        Generate the current state, next state, and the complex event label given the input action.
        For multi-label: returns binary vector [0,0,0,0,0,0,0,0,0,0] where position i-1 is 1 if event i fires.
        """
        assert self.fsm_list is not None
        # Initialize binary vector for 10 events (Events 1-10)
        ce_label_vector = np.zeros(10, dtype=np.int8)
        curr_states = []
        next_states = []

        for fsm in self.fsm_list:
            curr_states.append(fsm.get_current_state())
            l = fsm.update_state(input=action)
            next_states.append(fsm.get_current_state())

            # If this FSM fires (l > 0), set the corresponding position to 1
            # Event 1 -> index 0, Event 2 -> index 1, ..., Event 10 -> index 9
            if l > 0:
                ce_label_vector[l - 1] = 1

        self.event_label_sequence.append(ce_label_vector)
        self.state_input_sequence.append(curr_states)
        self.state_output_sequence.append(next_states)

    def _add_actions(self, action, window_length=None):
        """
        Add actions for a random or fixed number of windows and update time window elapsed
        """
        if window_length is not None:
            action_t = window_length
        else:
            # Convert seconds to windows, then sample random duration
            action_t_min, action_t_max = self._seconds_to_windows(self.action_length[action])
            action_t = np.random.randint(action_t_min, action_t_max + 1)
            
        action_id = self.label_mapping[action]

        for _ in range(action_t):
            self.action_sequence.append(action)
            # Randomly select an index from available samples for this action class
            chosen_idx = np.random.choice(self.class_index_list[action_id])
            self.sensor_data_indices.append(chosen_idx)  # Track the index for later data loading
            self.action_label_sequence.append(action_id)

            # Generate complex event label and states if FSMs are given
            if self.fsm_list is not None:
                self._add_complex_label(action=action)

        self.time_window_elapsed += action_t

    def get_data(self, dataset):
        # For index-based generation, we only need labels to know class distributions
        # We don't load the actual sensor data (sounds/imus) - that's loaded later from indices
        labels = dataset['labels']  # Shape: (N,)

        # Build class index list - maps each class ID to list of available sample indices
        class_index_list = []
        for i in range(self.nClasses):
            indices = np.where(labels == i)[0]
            class_index_list.append(indices)

        # Return None for data since we only track indices
        return None, class_index_list
    
    def _create_random_seq(self, action_list, activity_t_range, weights=None):
        """
        action_list: List of actions used for generate the random sequence of a specifc activity.
        activity_t_range: the min and max duration of the activity in SECONDS.
        weights: probabilities associated with each entry in action_list.
        """
        t_start = self.time_window_elapsed
        n_action = len(action_list)

        # Convert seconds to windows
        activity_t_min, activity_t_max = self._seconds_to_windows(activity_t_range)
        activity_t = np.random.randint(activity_t_min, activity_t_max + 1)

        while (self.time_window_elapsed - t_start) < activity_t: # Note that the generated seq may be slightly longer than total_t
            i = np.random.choice(a=n_action, size=1, p=weights)[0]
            action = action_list[i]
            self._add_actions(action)

        

class RestroomActivity(Activity):
    def __init__(self, data_cat, window_sec=5, enforce_window_length=None, fsm_list=None, simple_label=False, action_length_dict=None, activity_length_dict=None):
        """
        Args:
            data_cat (string): 'train' or 'test' dataset category.
            window_sec (float): window size in seconds (e.g., 2.5, 5).
            enforce_window_length (int): the fixed length of activity to generate.
            fsm_list (list): a list of FSM class instances.
            simple_label (bool): generate a single CE label if true, otherwise generate a list of CE labels corresponding to each timestamp.
        """
        self.action_length_dict = action_length_dict
        self.activity_length_dict = activity_length_dict
        super().__init__(name='restroom', data_cat=data_cat, window_sec=window_sec, fsm_list=fsm_list, simple_label=simple_label)
        self.enforce_window_length = enforce_window_length

    def _define_actions(self):
        """
        Define action/activity durations in SECONDS (not windows).
        window_sec determines conversion to windows.
        """
        self.action_length['walk'] = (5, 15)  # 5s to 15s
        self.action_length['sit'] = (10, 20)  # 10s to 20s
        self.action_length['flush_toilet'] = (5, 10)  # 5s to 10s
        self.action_length['wash'] = (5, 30)  # 5s to 30s
        self.action_length['type'] = (10, 15)  # 10s to 15s
        self.action_length['click_mouse'] = (5, 5)  # 5s

        if self.action_length_dict is not None:
            for action, action_len in self.action_length_dict.items():
                self.action_length[action] = action_len

        # For activities with random repeating pattern
        self.activity_length['restroom_sit'] = (10, 120)  # 10s to 2min
        self.activity_length['work'] = (15, 180)  # 15s to 3min
        self.activity_length['walk_in'] = (5, 20)  # 5s to 20s
        self.activity_length['wander'] = (20, 180)  # 20s to 3min

        if self.activity_length_dict is not None:
            for activity, activity_len in self.activity_length_dict.items():
                self.activity_length[activity] = activity_len

    def generate_activity(self):
        """
        Synthesize the restroom activity:
            - walk1 -> wash1? -> (sit + flush + walk2? + wash?)? -> wander -> work -> (walk) -> work -> (walk) ('?' means a random action that may not happen)
        """
        ## cases when you walk for some time and wash, or after flush directly go back to work
        use_restroom = 0.8
        wash_prob1 = 0.2
        wash_prob2 = 0.8
        walk_prob = 0.9
        break_prob = 0.3

        use_restroom_flag = False

        # walk action (walk in)
        action_list = ['walk']
        self._create_random_seq(action_list, self.activity_length['walk_in'])
        
        # wash action (wash hands - may not happen)
        if np.random.rand() < wash_prob1:
            self._add_actions('wash')

        if np.random.rand() < use_restroom: # Use restroom
            use_restroom_flag = True

            # sit action (sit on toilet)
            self._create_random_seq(['sit'], self.activity_length['restroom_sit'])

            # flush action (flush the toilet)
            self._add_actions('flush_toilet')

            # walk action
            if np.random.rand() < walk_prob:
                self._add_actions('walk', window_length=1)

            # wash action (wash hands - may not happen)
            if np.random.rand() < wash_prob2:
                self._add_actions('wash')

        # walk action (walk away)
        action_list = ['walk', 'sit']
        # if not use_restroom_flag:
        #     self.activity_length['wander'] = (4, 48) # 20s - 4min if didn't use restroom
        self._create_random_seq(action_list, self.activity_length['wander'], weights=[0.8, 0.2])

        # Start working
        action_list = ['sit', 'type', 'click_mouse']
        self._create_random_seq(action_list, self.activity_length['work'], weights=[0.3, 0.3, 0.4])
        
        # May take a break
        if np.random.rand() < break_prob:
            self._add_actions('walk')

        # Start working
        action_list = ['sit', 'type', 'click_mouse']
        self._create_random_seq(action_list, self.activity_length['work'], weights=[0.3, 0.3, 0.4])
        
        # May take a break
        if np.random.rand() < break_prob:
            self._add_actions('walk')
    
        # Generate sequence of fixed length
        if self.enforce_window_length is not None:
            # Truncate the sequence
            if self.time_window_elapsed > self.enforce_window_length:
                self._truncate_events(self.enforce_window_length)
            # Extend the sequence with the last action
            elif self.time_window_elapsed < self.enforce_window_length:
                add_window_length = self.enforce_window_length - self.time_window_elapsed
                self._add_actions('walk', window_length=add_window_length)

            self.time_window_elapsed = len(self.action_sequence)

        return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed
    

# class WalkingActivity(Activity):
#     def __init__(self, data_cat):
#         super().__init__(name='walk_only', data_cat=data_cat)

#     def _define_actions(self):
#         """
#         window size =  5s
#         """
#         self.action_length['walk'] = (1, 180) # 5s - 15min

#     def generate_activity(self):
#         """
#         Synthesize the walking only activity:
#             - walk
#         """

#         self._add_actions('walk')

#         return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed
    

# class SittingActivity(Activity):
#     def __init__(self, data_cat):
#         super().__init__(name='sit_only', data_cat=data_cat)

#     def _define_actions(self):
#         """
#         window size =  5s
#         """
#         self.action_length['sit'] = (60, 360) # 5min - 30min

#     def generate_activity(self):
#         """
#         Synthesize the sitting still activity:
#             - sit
#         """
#         self._add_actions('sit')

#         return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed
    

# class WorkingActivity(Activity):
#     def __init__(self, data_cat):
#         self.action_probs = {}
#         super().__init__(name='work', data_cat=data_cat)

#     def _define_actions(self):
#         """
#         window size =  5s
#         """
#         self.action_length['sit'] = (1, 60) # 5s - 5min
#         self.action_length['type'] = (1, 4) # 5s - 20s
#         self.action_length['click_mouse'] = (1, 4) # 5s - 20s
#         self.action_length['drink'] = (1, 3) # 5si]s - 15s

#         self.action_probs['sit'] = 0.32
#         self.action_probs['type'] = 0.32
#         self.action_probs['click_mouse'] = 0.32
#         self.action_probs['drink'] = 0.04
#         assert sum(self.action_probs.values()) == 1


#     def generate_activity(self):
#         """
#         Synthesize the working activity:
#             - randomly switch between sit, type, and click mouse within a given time interval 'totoal_t'
#         """
#         sit_prob = self.action_probs['sit']
#         type_prob = self.action_probs['type']
#         click_prob = self.action_probs['click_mouse']
#         drink_prob = self.action_probs['drink']

#         total_t = np.random.randint(360, 1440 + 1) # 30min - 2h

#         while self.time_window_elapsed < total_t:
#             prob = np.random.rand()
#             if prob < sit_prob: 
#                 # sit happens
#                 self._add_actions('sit')

#             elif prob < sit_prob + type_prob: 
#                 # type happens
#                 self._add_actions('type')

#             elif prob < sit_prob + type_prob + click_prob: 
#                 # click happens
#                 self._add_actions('click')

#             else: 
#                 # drink happens
#                 self._add_actions('drink')

#         return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed
    

# class DrinkingActivity(Activity):
#     def __init__(self, data_cat):
#         super().__init__(name='drink_only', data_cat=data_cat)

#     def _define_actions(self):
#         """
#         window size =  5s
#         """
#         self.action_length['drink'] = (1, 3) # 5s - 15s

#     def generate_activity(self):
#         """
#         Synthesize the drinking-only activity:
#             - sit
#         """
#         self._add_actions('drink')

#         return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed
    

class OralCleaningActivity(Activity):
    def __init__(self, data_cat, window_sec=5, enforce_window_length=None, action_length={}, fsm_list=None, simple_label=False, action_length_dict=None, activity_length_dict=None):
        """
        Args:
            data_cat (string): 'train' or 'test' dataset category.
            window_sec (float): window size in seconds (e.g., 2.5, 5).
            enforce_window_length (int): the fixed length of activity to generate.
            action_length (tuple, dict): key - activity name, value - (min_time, max_time)
            fsm_list (list): a list of FSM class instances.
            simple_label (bool): generate a single CE label if true, otherwise generate a list of CE labels corresponding to each timestamp.
        """
        self.action_length_dict = action_length_dict
        self.activity_length_dict = activity_length_dict
        super().__init__(name='oral_clean', data_cat=data_cat, window_sec=window_sec, fsm_list=fsm_list, simple_label=simple_label)
        self.enforce_window_length = enforce_window_length
        if action_length: # if action_length is not empty
            self.action_length = action_length

    def _define_actions(self):
        """
        Define action/activity durations in SECONDS (not windows).
        """
        self.action_length['walk'] = (5, 60)  # 5s to 1min
        self.action_length['wash'] = (5, 30)  # 5s to 30s
        self.action_length['brush_teeth'] = (15, 180)  # 15s to 3min
        self.action_length['eat'] = (5, 10)  # 5s to 10s
        self.action_length['sit'] = (5, 10)  # 5s to 10s
        self.action_length['drink'] = (5, 15)  # 5s to 15s

        if self.action_length_dict is not None:
            for action, action_len in self.action_length_dict.items():
                self.action_length[action] = action_len

        # For activities with random repeating pattern
        self.activity_length['meal'] = (20, 60)  # 20s to 1min
        self.activity_length['sit_session'] = (15, 60)  # 15s to 1min

        if self.activity_length_dict is not None:
            for activity, activity_len in self.activity_length_dict.items():
                self.activity_length[activity] = activity_len

    def generate_activity(self):
        """
        Synthesize the oral cleaning activity:
            - sit? -> walk1 -> wash1? -> brush -> wash2 -> (wash + brush)? -> walk2 ('?' means a random action that may not happen) -> drink? -> meal -> sit
        """
        wash_prob = 0.4
        brush_again_prob = 0.3
        sit_session_prob = 0.3
        meal_prob = 0.6
        drink_prob = 0.1
        walk_prob = 0.2

        if np.random.rand() < sit_session_prob:
            # Sit for some time
            self._create_random_seq(['sit'], self.activity_length['sit_session'])

        # walk action (walk in)
        self._add_actions('walk')

        # wash action (wash before brushing teeth - may not happen)
        if np.random.rand() < wash_prob:
            self._add_actions('wash')

        # brush_teeth action
        self._add_actions('brush_teeth')

        temp = self.action_length['wash']
        # May wash + brush again
        if np.random.rand() < brush_again_prob:
            self.action_length['wash'] = (5, 10)  # 5s to 10s
            self._add_actions('wash')

            self.action_length['brush_teeth'] = (5, 15)  # 5s to 15s
            self._add_actions('brush_teeth')

        # wash action after brushing teeth
        self.action_length['wash'] = temp
        self._add_actions('wash')

        # walk action (walk away)
        self._add_actions('walk')

        if np.random.rand() < drink_prob:
            self._add_actions('drink')

        if np.random.rand() < walk_prob:
            # walk action (walk to eat)
            self._add_actions('walk')

        if np.random.rand() < meal_prob:
            self._create_random_seq(['sit', 'eat', 'drink'], self.activity_length['meal'], weights=[0.3, 0.5, 0.2])


        # Generate sequence of fixed length
        if self.enforce_window_length is not None:
            # Truncate the sequence
            if self.time_window_elapsed >= self.enforce_window_length:
                self._truncate_events(self.enforce_window_length)
            # Extend the sequence with the last action
            else:
                add_window_length = self.enforce_window_length - self.time_window_elapsed
                self._add_actions('sit', window_length=add_window_length)

            self.time_window_elapsed = len(self.action_sequence)

        return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed


class HavingMealActivity(Activity):
    def __init__(self, data_cat, window_sec=5, enforce_window_length=None, fsm_list=None, simple_label=False, action_length_dict=None, activity_length_dict=None):
        """
        Args:
            data_cat (string): 'train' or 'test' dataset category.
            window_sec (float): window size in seconds (e.g., 2.5, 5).
            enforce_window_length (int): the fixed length of activity to generate.
            fsm_list (list): a list of FSM class instances.
            simple_label (bool): generate a single CE label if true, otherwise generate a list of CE labels corresponding to each timestamp.
        """
        self.action_length_dict = action_length_dict
        self.activity_length_dict = activity_length_dict
        super().__init__(name='have_meal', data_cat=data_cat, window_sec=window_sec, fsm_list=fsm_list, simple_label=simple_label)
        self.enforce_window_length = enforce_window_length

        

    def _define_actions(self):
        """
        Define action/activity durations in SECONDS (not windows).
        """
        self.action_length['walk'] = (5, 20)  # 5s to 20s
        self.action_length['wash'] = (5, 30)  # 5s to 30s
        self.action_length['eat'] = (5, 10)  # 5s to 10s
        self.action_length['sit'] = (5, 10)  # 5s to 10s
        self.action_length['drink'] = (5, 15)  # 5s to 15s
        self.action_length['flush_toilet'] = (5, 10)  # 5s to 10s
        self.action_length['click_mouse'] = (5, 5)  # 5s
        self.action_length['type'] = (10, 15)  # 10s to 15s

        if self.action_length_dict is not None:
            for action, action_len in self.action_length_dict.items():
                self.action_length[action] = action_len

        # For activities with random repeating pattern
        self.activity_length['work'] = (15, 240)  # 15s to 4min
        self.activity_length['meal'] = (15, 30)  # 15s to 30s
        self.activity_length['meal2'] = (15, 30)  # 15s to 30s
        self.activity_length['restroom_sit'] = (10, 20)  # 10s to 20s
        self.activity_length['after_meal_rest'] = (120, 240)  # 2min to 4min

        if self.activity_length_dict is not None:
            for activity, activity_len in self.activity_length_dict.items():
                self.activity_length[activity] = activity_len


    def generate_activity(self):
        """
        Synthesize the having meal activity:
            - wash? -> walk -> (restroom or work)? -> wash? -> meal -> walk2? -> wash? -> meal ('?' means a random action that may not happen)
        """
        wash_prob = 0.6
        other_prob = 0.3
        walk_prob = 0.3
        meal_again_prob = 0.3
        work_again_prob = 0.5
        rest_prob = 0.9

        # wash action (wash before having meal - may not happen)
        if np.random.rand() < wash_prob:
            self._add_actions('wash')

        # walk action (walk in)
        self._add_actions('walk')

        if np.random.rand() < other_prob: # Touch other objects
            if np.random.rand() < 1/2: # Use restroom
                action_list = ['sit']
                self._create_random_seq(action_list, self.activity_length['restroom_sit'])
                self._add_actions('flush_toilet')
            else: # Back to work
                action_list = ['sit', 'type', 'click_mouse']
                # ORIGINAL (5s windows): (3, 12)  # 15s - 60s
                self._create_random_seq(action_list, (15, 60), weights=[0.2, 0.3, 0.5])  # 15s to 1min
            
        # May wash again
        if np.random.rand() < wash_prob:
            self._add_actions('wash')

        # have meals
        action_list = ['sit', 'eat', 'drink']
        self._create_random_seq(action_list, self.activity_length['meal'], weights=[0.3, 0.5, 0.2])

        if np.random.rand() < walk_prob:
            self._add_actions('walk')

        if np.random.rand() < wash_prob:
            self._add_actions('wash')

        # may have meals again 
        if np.random.rand() < meal_again_prob:
            action_list = ['sit', 'eat', 'drink']
            self._create_random_seq(action_list, self.activity_length['meal2'], weights=[0.3, 0.3, 0.4])

        # may rest
        if np.random.rand() < rest_prob:
            self._create_random_seq(['sit', 'walk'], self.activity_length['after_meal_rest'], weights=[0.7, 0.3])


        # may go back to work
        if np.random.rand() < work_again_prob:
            self._add_actions('walk')
            self._create_random_seq(['sit', 'type', 'click_mouse'], self.activity_length['work'], weights=[0.2, 0.4, 0.4])

        if np.random.rand() < work_again_prob:
            self._add_actions('walk')
            self._create_random_seq(['sit', 'type', 'click_mouse'], self.activity_length['work'], weights=[0.2, 0.4, 0.4])

        # Generate sequence of fixed length
        if self.enforce_window_length is not None:
            # Truncate the sequence
            if self.time_window_elapsed >= self.enforce_window_length:
                self._truncate_events(self.enforce_window_length)
            # Extend the sequence with the last action
            else:
                add_window_length = self.enforce_window_length - self.time_window_elapsed
                self._add_actions('walk', window_length=add_window_length)

            self.time_window_elapsed = len(self.action_sequence)

        return self.action_sequence, self.sensor_data_indices, self.action_label_sequence, self.time_window_elapsed