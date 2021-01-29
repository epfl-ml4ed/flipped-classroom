#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from helper.dataset.data_preparation import count_actions

class FrequencyAction(Feature):

    def __init__(self, data, settings):
        super().__init__('frequency_action' + ('_' + settings['type'].lower().replace('.', '_') if 'type' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        if 'type' in self.settings:
            return count_actions(self.data, self.settings['type']) / len(self.data.index)
        udata = self.data.merge(self.settings['course'].get_schedule()[['id', 'duration']], left_on=['video_id'], right_on=['id'])
        udata.loc[:, 'day'] = udata.loc[:, 'date'].dt.date  # Create column with the date but not the time
        udata.drop_duplicates(subset=['video_id', 'day'], inplace=True)  # Only keep on event per video per day
        watching_time = udata['duration'].sum() / 3600  # hours
        return len(udata.index) / watching_time if watching_time != 0 else 0
