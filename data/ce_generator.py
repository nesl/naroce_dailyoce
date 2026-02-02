import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from stages import *
from activities import *


def convert_time(time):
        t = datetime.strptime(time, "%H:%M").time()
        t_second = (t.hour * 60 + t.minute) * 60 + t.second
        return t_second


class CEGenerator():
    def __init__(self, n_data, start_time, end_time, time_unit=5):
        """
        Args:
        """
        self.start_time = convert_time(start_time)
        self.end_time = convert_time(end_time)
        self.n_data = n_data
        self.time_unit = time_unit # This is the window size of multimodal dataset we are going to use

        self.total_time_window = int((self.end_time - self.start_time) // self.time_unit)
        # self.time_window_elapsed = 0
        self.stage_list = []
        self.time_seires = []
        
    def generate_events(self):
        raise NotImplementedError

    # def _get_stage(self, t):
    #     raise NotImplementedError


    def _define_stages(self):
        """
        Define time period of each stage.
        # """
        # self.stage_list.append(DaytimeStage(["06:00", "18:00"])) # default stage before 18:00
        # DailyCareStage(["06:00", "09:00"]) # happen at most 1 time in the time period
        # MealStage(["11:00", "13:00"]) # happen at most 1 time in the time period
        # MealStage(["18:00", "20:00"]) # happen at most 1 time in the time period
        # EveningStage(["18:00","23:00"]) # (need to be after meal), default stage after 18:00
        # DailyCareStage(["20:00", "23:00"]) # happen at most 1 time in the time period


class CE5min(CEGenerator):
    """
    Generate events of only 5 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, data_cat, start_time="00:00", end_time="00:05", time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            data_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data, start_time, end_time, time_unit)
        self.data_cat = data_cat
        self.simple_label = simple_label
        
    def generate_event(self, scenario_id):
        """
        x,y, y[t] = a inidates the event a ends at time t
        """
        t = self.start_time
        total_time_window = self.total_time_window
        data_cat = self.data_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        fsm_list = [Event1FSM(window_sec=self.time_unit), Event2FSM(window_sec=self.time_unit), Event3FSM(window_sec=self.time_unit), Event4FSM(window_sec=self.time_unit), Event5FSM(window_sec=self.time_unit),\
                    Event6FSM(window_sec=self.time_unit), Event7FSM(window_sec=self.time_unit), Event8FSM(window_sec=self.time_unit), Event9FSM(window_sec=self.time_unit), Event10FSM(window_sec=self.time_unit)]

        if scenario_id == 0:
            #  Scenario 1
            restroom_activity = RestroomActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        elif scenario_id == 1:
            #  Scenario 2
            meal_activity = HavingMealActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        elif scenario_id == 2:
            #  Scenario 3
            oral_activity = OralCleaningActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("scenario_id out of range.")

        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
        
    def generate_CE_dataset(self, is_positive_sample=False, state_mapping=None):
        ce_data = []
        ce_labels = []
        actions = []
        ae_labels = []

        # n_event = 3
        n_scenario = 3
        n_data_per_event = self.n_data // n_scenario

        for i in range(n_scenario):
            ce_data_temp = []
            ce_labels_temp = []
            actions_temp = []
            ae_labels_temp = []
            if i == n_scenario - 1:
                n = self.n_data - i * n_data_per_event
            else:
                n = n_data_per_event

            n_zero_data_per_event = n // 2
            n_zero_count = 0

            while len(ce_data_temp) < n:

                data_sequence, event_label_sequence, _, _, action_sequence, action_label_sequence, _, _ = self.generate_event(i)

                # For multi-label: event_label_sequence is list of binary vectors
                # Stack into (n_windows, 10) array
                event_label_array = np.stack(event_label_sequence, axis=0)  # (n_windows, 10)

                # Check for positive samples: any event fires at any time window
                if is_positive_sample and np.sum(event_label_array) == 0:
                    continue

                data_sequence = np.concatenate([x[None, ...] for x in data_sequence], axis=0)
                # Note: No label adjustment needed - binary vectors are already 0-indexed
                ce_data_temp.append(data_sequence)
                ce_labels_temp.append(event_label_array)

                action_sequence = np.array(action_sequence)
                actions_temp.append(action_sequence)

                action_label_sequence = np.array(action_label_sequence)
                ae_labels_temp.append(action_label_sequence)

            ce_data.extend(ce_data_temp)
            ce_labels.extend(ce_labels_temp)
            actions.extend(actions_temp)
            ae_labels.extend(ae_labels_temp)
        
        ce_data = np.concatenate([x[None, ...] for x in ce_data], axis=0)
        ce_labels = np.stack(ce_labels, axis=0)  # Shape: (n_samples, n_windows, 10)
        actions = np.stack(actions, axis=0)
        ae_labels = np.stack(ae_labels, axis=0)

        return ce_data, ce_labels, ae_labels, actions

    def generate_CE_index_dataset(self, is_positive_sample=False):
        """
        Generate lightweight index dataset instead of full sensor data.

        Returns:
            dict containing:
                - sensor_data_indices: List of index arrays (one per CE sample)
                - data_sources: 'train' or 'test' for each sample
                - scenario_ids: Which scenario (0=restroom, 1=meal, 2=oral)
                - ce_labels: Complex event label sequences
                - actions: Action name sequences
                - action_labels: Action label ID sequences
                - config: Generation parameters
        """
        index_records = []
        ce_labels = []
        actions = []
        ae_labels = []

        n_scenario = 3
        n_data_per_scenario = self.n_data // n_scenario

        # Create progress bar for overall progress
        pbar = tqdm(total=self.n_data, desc="Generating CE samples", unit="sample")

        for i in range(n_scenario):
            index_records_temp = []
            ce_labels_temp = []
            actions_temp = []
            ae_labels_temp = []

            if i == n_scenario - 1:
                n = self.n_data - i * n_data_per_scenario
            else:
                n = n_data_per_scenario

            while len(index_records_temp) < n:
                # Generate event - data_sequence now contains sensor_data_indices
                sensor_data_indices, event_label_sequence, _, _, action_sequence, action_label_sequence, _, _ = self.generate_event(i)

                # For multi-label: event_label_sequence is list of binary vectors
                # Stack into (n_windows, 10) array
                event_label_array = np.stack(event_label_sequence, axis=0)  # (n_windows, 10)

                # Check for positive samples: any event fires at any time window
                if is_positive_sample and np.sum(event_label_array) == 0:
                    continue

                # Store index record using data directly from generate_event
                record = {
                    'scenario_id': i,
                    'sensor_data_indices': sensor_data_indices,
                    'data_source': self.data_cat,
                }
                index_records_temp.append(record)

                # Note: No label adjustment needed - binary vectors are already 0-indexed
                ce_labels_temp.append(event_label_array)

                action_sequence = np.array(action_sequence)
                actions_temp.append(action_sequence)

                action_label_sequence = np.array(action_label_sequence)
                ae_labels_temp.append(action_label_sequence)

                # Update progress bar
                pbar.update(1)

            index_records.extend(index_records_temp)
            ce_labels.extend(ce_labels_temp)
            actions.extend(actions_temp)
            ae_labels.extend(ae_labels_temp)

        pbar.close()

        ce_labels = np.stack(ce_labels, axis=0)
        actions = np.stack(actions, axis=0)
        ae_labels = np.stack(ae_labels, axis=0)

        return {
            'index_records': index_records,
            'ce_labels': ce_labels,
            'actions': actions,
            'ae_labels': ae_labels,
            'config': {
                'time_window_class': self.__class__.__name__,
                'n_data': self.n_data,
                'data_cat': self.data_cat,
                'time_unit': self.time_unit,
                'start_time': f"{self.start_time//3600:02d}:{(self.start_time%3600)//60:02d}",
                'end_time': f"{self.end_time//3600:02d}:{(self.end_time%3600)//60:02d}",
                'simple_label': self.simple_label,
            }
        }

    def save_index_dataset(self, output_dir='CE_dataset', is_positive_sample=False, suffix='', split_name=None):
        """
        Save index dataset to compressed npz file with n_samples in filename.

        Args:
            output_dir: Directory to save the index file (default: 'CE_dataset')
            is_positive_sample: If True, only generate samples with violations
            suffix: Optional suffix for filename (e.g., '_rule' for rule-only test sets)
            split_name: Dataset split name for filename ('train', 'val', 'test'). If None, uses self.data_cat

        Returns:
            filepath: Path to the saved index file
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with time window class, split name, suffix, and n_samples
        # e.g., ce5min_train_indices_2000.npz, ce5min_val_indices.npz, ce5min_test_rule_indices.npz
        class_name = self.__class__.__name__.lower()  # ce5min, ce15min, etc.

        # Use split_name if provided, otherwise fall back to data_cat
        file_split = split_name if split_name is not None else self.data_cat

        # Train split includes n_samples in filename, val/test do not
        if file_split == 'train':
            filename = f"{class_name}_{file_split}{suffix}_indices_{self.n_data}.npz"
        else:
            filename = f"{class_name}_{file_split}{suffix}_indices.npz"
        filepath = os.path.join(output_dir, filename)

        index_data = self.generate_CE_index_dataset(is_positive_sample=is_positive_sample)

        # Extract data from records
        sensor_indices = [record['sensor_data_indices'] for record in index_data['index_records']]
        scenario_ids = [record['scenario_id'] for record in index_data['index_records']]
        data_sources = [record['data_source'] for record in index_data['index_records']]

        print(f"Saving to {filepath}...")
        np.savez_compressed(
            filepath,
            sensor_data_indices=np.array(sensor_indices, dtype=object),
            scenario_ids=np.array(scenario_ids),
            data_sources=np.array(data_sources),
            ce_labels=index_data['ce_labels'],
            actions=index_data['actions'],
            action_labels=index_data['ae_labels'],
            config=json.dumps(index_data['config'])
        )

        # Print statistics
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ Saved index dataset: {filepath}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Samples: {len(sensor_indices)}")
        print(f"  Avg indices per sample: {np.mean([len(idx) for idx in sensor_indices]):.1f}")

        return filepath

class CE15min(CE5min):
    """
    Generate events of only 5 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, data_cat, time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            data_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data=n_data, data_cat=data_cat, start_time="00:00", end_time="00:15", time_unit=time_unit, simple_label=simple_label)
        self.data_cat = data_cat
        self.simple_label = simple_label
        
    def generate_event(self, scenario_id):
        """
        x,y, y[t] = a inidates the event a ends at time t
        """
        t = self.start_time
        total_time_window = self.total_time_window
        data_cat = self.data_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        fsm_list = [Event1FSM(window_sec=self.time_unit), Event2FSM(window_sec=self.time_unit), Event3FSM(window_sec=self.time_unit), Event4FSM(window_sec=self.time_unit), Event5FSM(window_sec=self.time_unit),\
                    Event6FSM(window_sec=self.time_unit), Event7FSM(window_sec=self.time_unit), Event8FSM(window_sec=self.time_unit), Event9FSM(window_sec=self.time_unit), Event10FSM(window_sec=self.time_unit)]

        if scenario_id == 0:
            #  Scenario 1
            restroom_action_length_dict = {
                                        #    'walk': (5, 15), # 5s - 15s
                                        #    'sit': (10, 20),  # 10s - 20s
                                           'wash': (5, 60) # 5s - 1min
                                           }
            restroom_activity_length_dict = {'work': (180, 600), # 3min - 10min
                                             'restroom_sit': (10, 300),  # 10s - 5min
                                             'wander': (120, 420), # 2min - 7min # in order to make temporal span between 5min to 8min (may use restrrom for at most 45s before wandering)
                                             'walk_in': (60, 120), # 1min - 2min
                                            }
            restroom_activity = RestroomActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label, action_length_dict=restroom_action_length_dict, activity_length_dict=restroom_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        elif scenario_id == 1:
            #  Scenario 2
            meal_action_length_dict = {'walk': (15, 120), # 15s - 2min
                                       'wash': (5, 60), # 5s - 1min
                                        }
            meal_activity_length_dict = {'work': (180, 300), # 3min - 5min
                                         'meal': (180, 300), # 3min - 5min
                                         'after_meal_rest': (120, 480) # 2min - 8min
                                        }
            meal_activity = HavingMealActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label, action_length_dict=meal_action_length_dict, activity_length_dict=meal_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        elif scenario_id == 2:
            #  Scenario 3
            oral_action_length_dict = {'walk': (15, 180), # 15s - 3min
                                       'wash': (5, 60), # 5s - 1min
                                       'brush_teeth': (15, 240), # 15s - 4min
                                    #    'sit': (60, 300), # 1min - 5min
                                       }
            oral_activity_length_dict = {
                                         'meal': (180, 300), # 3min - 5min
                                         'sit_session': (20, 180) # 20s - 3min
                                        }
            oral_activity = OralCleaningActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label, action_length_dict=oral_action_length_dict, activity_length_dict=oral_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("scenario_id out of range.")
        
        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
        

class CE3min(CE5min):
    """
    Generate events of only 5 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, data_cat, time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            data_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data=n_data, data_cat=data_cat, start_time="00:00", end_time="00:03", time_unit=time_unit, simple_label=simple_label)
        self.data_cat = data_cat
        self.simple_label = simple_label
        
    # def generate_event(self, scenario_id):
    #     """
    #     x,y, y[t] = a inidates the event a ends at time t
    #     """
    #     t = self.start_time
    #     total_time_window = self.total_time_window
    #     data_cat = self.data_cat
    #     time_window_elapsed = 0

    #     data_sequence =[]
    #     event_label_sequence = []
    #     state_input_sequence = []
    #     state_output_sequence = []
    #     action_sequence = []
    #     action_label_sequence = []

    #     if scenario_id == 0:
    #         #  Scenario 1: walk1 -> wash1? -> sitting -> flush -> wash2? -> walk2 ('?' means a random action that may not happen)
    #         restroom_action_length_dict = {'walk': (5, 10), # 5s - 10s
    #                                        'sit': (10, 15), # 10s - 15s
    #                                        }
    #         restroom_activity_length_dict = {'work': (30, 60), # 30s - 1min
    #                                          'wander': (30, 120), # 30s - 2min
    #                                          'walk_in': (5, 15), # 5s - 15s
    #                                         }
    #         event1_fsm = Event1FSM(window_sec=self.time_unit)
    #         restroom_activity = RestroomActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=[event1_fsm], simple_label=self.simple_label, action_length_dict=restroom_action_length_dict, activity_length_dict=restroom_activity_length_dict)
    #         action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
    #         event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
    #         time_window_elapsed += time_window
    #         t += time_window_elapsed * self.time_unit

    #     elif scenario_id == 1:
    #         #  Scenario 2: wash? -> walk -> (restroom or work)? -> wash? -> meal -> walk2? -> wash? -> meal
    #         meal_action_length_dict = {'walk': (5, 15), # 5s - 15s
    #                                    }
    #         meal_activity_length_dict = {'work': (30, 60), # 30s - 1min
    #                                      'meal': (30, 60), # 30s - 1min
    #                                     }
    #         event2_fsm = Event2FSM(window_sec=self.time_unit)
    #         meal_activity = HavingMealActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=[event2_fsm], simple_label=self.simple_label, action_length_dict=meal_action_length_dict, activity_length_dict=meal_activity_length_dict)
    #         action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
    #         event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
    #         time_window_elapsed += time_window
    #         t += time_window_elapsed * self.time_unit

    #     elif scenario_id == 2:
    #         #  Scenario 3: sit? -> walk1 -> wash1? -> brush -> wash2 -> (wash + brush)? -> walk2 ('?' means a random action that may not happen)
    #         oral_action_length_dict = {'walk': (5, 10), # 5s - 10s
    #                                    'wash': (5, 15), # 5s - 15s
    #                                    'brush_teeth': (15, 180), # 15s - 3min
    #                                    'sit': (5, 10), # 5s - 10s
    #                                    }
    #         event3_fsm = Event3FSM(window_sec=self.time_unit)
    #         oral_activity = OralCleaningActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=[event3_fsm], simple_label=self.simple_label, action_length_dict=oral_action_length_dict)
    #         action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
    #         event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
    #         time_window_elapsed += time_window
    #         t += time_window_elapsed * self.time_unit

    #     else:
    #         raise Exception("scenario_id out of range.")
        
    #     return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t


class CE30min(CE5min):
    """
    Generate events of 30 minutes. 
    The absolute time doesn't matter in this case.
    """
    def __init__(self, n_data, data_cat, time_unit=5, simple_label=False):
        """
        Args:
            n_data (int): number of data samples to generate.
            data_cat (string): 'train' or 'test' dataset category.
        """
        super().__init__(n_data=n_data, data_cat=data_cat, start_time="00:00", end_time="00:30", time_unit=time_unit, simple_label=simple_label)
        self.data_cat = data_cat
        self.simple_label = simple_label
        
    def generate_event(self, scenario_id):
        """
        x,y, y[t] = a inidates the event a ends at time t
        """
        t = self.start_time
        total_time_window = self.total_time_window
        data_cat = self.data_cat
        time_window_elapsed = 0

        data_sequence =[]
        event_label_sequence = []
        state_input_sequence = []
        state_output_sequence = []
        action_sequence = []
        action_label_sequence = []

        fsm_list = [Event1FSM(window_sec=self.time_unit), Event2FSM(window_sec=self.time_unit), Event3FSM(window_sec=self.time_unit), Event4FSM(window_sec=self.time_unit), Event5FSM(window_sec=self.time_unit),\
                    Event6FSM(window_sec=self.time_unit), Event7FSM(window_sec=self.time_unit), Event8FSM(window_sec=self.time_unit), Event9FSM(window_sec=self.time_unit), Event10FSM(window_sec=self.time_unit)]

        if scenario_id == 0:
            #  Scenario 1
            restroom_action_length_dict = {
                                        #    'walk': (5, 15), # 5s - 15s
                                        #    'sit': (10, 20),  # 10s - 20s
                                           'wash': (5, 60) # 5s - 1min
                                           }
            restroom_activity_length_dict = {'work': (300, 900), # 5min - 15min
                                             'restroom_sit': (10, 480),  # 10s - 8min
                                             'wander': (300, 900), # 5min - 15min # in order to make temporal span between 5min to 8min (may use restrrom for at most 45s before wandering)
                                             'walk_in': (180, 300), # 3min - 5min
                                            }
            restroom_activity = RestroomActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label, action_length_dict=restroom_action_length_dict, activity_length_dict=restroom_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = restroom_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = restroom_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        elif scenario_id == 1:
            #  Scenario 2
            meal_action_length_dict = {'walk': (30, 300), # 30s - 5min
                                       'wash': (5, 120), # 5s - 2min
                                        }
            meal_activity_length_dict = {'work': (300, 600), # 5min - 10min
                                         'meal': (300, 720), # 5min - 12min
                                         'after_meal_rest': (120, 720) # 2min - 12min
                                        }
            meal_activity = HavingMealActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label, action_length_dict=meal_action_length_dict, activity_length_dict=meal_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = meal_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = meal_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        elif scenario_id == 2:
            #  Scenario 3
            oral_action_length_dict = {'walk': (15, 300), # 15s - 5min
                                       'wash': (5, 120), # 5s - 2min
                                       'brush_teeth': (15, 240), # 15s - 4min
                                    #    'sit': (60, 300), # 1min - 5min
                                       }
            oral_activity_length_dict = {
                                         'meal': (180, 600), # 3min - 10min
                                         'sit_session': (30, 360) # 30s - 6min
                                        }
            oral_activity = OralCleaningActivity(data_cat, window_sec=self.time_unit, enforce_window_length=total_time_window, fsm_list=fsm_list, simple_label=self.simple_label, action_length_dict=oral_action_length_dict, activity_length_dict=oral_activity_length_dict)
            action_sequence, data_sequence, action_label_sequence, time_window = oral_activity.generate_activity()
            event_label_sequence, state_input_sequence, state_output_sequence = oral_activity.generate_complex_label()
            time_window_elapsed += time_window
            t += time_window_elapsed * self.time_unit

        else:
            raise Exception("scenario_id out of range.")
        
        return data_sequence, event_label_sequence, state_input_sequence, state_output_sequence, action_sequence, action_label_sequence, time_window_elapsed, t
        



def count_arrays(arrays):
    # Initialize counters for 1 to 10 and "only 0"
    counts = {i: 0 for i in range(11)}
    
    for array in arrays:
        unique_elements = set(array)  # Get the unique elements in the array
        
        # Check for "only 0"
        if unique_elements == {0}:
            counts[0] += 1
        
        # Count arrays containing integers 1 through 10
        for i in range(1, 11):
            if i in unique_elements:
                counts[i] += 1
    
    return counts

if __name__ == '__main__':
    n_data = 2000
    ce5 = CE15min(n_data, 'train')
    # action_data, event_labels, in_states, out_states, actions, action_labels, windows, t = ce5.generate_event(0)
    # # for a,l in zip(actions, labels):
    # #     print(a,l)

    # print(event_labels)
    # # print(in_states)
    # # print(out_states)
    # print(actions)
    # print(action_labels)

    # print(len(action_data))
    # print(len(event_labels))
    # # print(len(in_states))
    # # print(len(out_states))
    # print(len(actions))
    # print(len(action_labels))

    # print(windows)
    # print(t)

    ce_data, ce_labels, ae_labels, actions = ce5.generate_CE_dataset()
    print(ce_data.shape, ce_labels.shape, ae_labels.shape)
    print(ce_labels[np.random.randint(n_data)])
    # for i in range(len(ce_data)):
    #     print(f"Index {i}:")
    #     print(actions[i])
    #     print(ce_labels[i])
    print(count_arrays(ce_labels))
    


