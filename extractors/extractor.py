from helpers.feature_extraction import *

class Extractor():

    def __init__(self, name='base'):
        self.name = name

    def getName(self):
        return self.name

    def totalActions(self, udata):
        """Counts the total number of actions performed across every videos"""
        return len(udata)

    def totalEvents(self, udata, event_type, unique=False):
        edata = udata[udata['EventType'] == event_type]
        return len(edata) if not unique else len(edata.drop_duplicates(subset=['ElementID'], keep='first'))

    def totalViews(self, udata):
        """ 
        Counts the total of videos views (rewatch and interruption included)
        Assumption: consider that a video is watched at most once per day
        """
        copy = udata.copy()
        copy['Day'] = udata.Date.dt.date
        # From the assumption the video view is a unique pair (video id, day)
        return len(copy.drop_duplicates(subset=['VideoID', 'Day']))
    
    def avgWeeklyPropWatched(self, udata):
        return weekly_prop_watched(udata).mean()

    def stdWeeklyPropWatched(self, udata):
        return weekly_prop_watched(udata).std()
    
    def avgWeeklyPropReplayed(self, udata):
        return weekly_prop_replayed(udata).mean()

    def stdWeeklyPropReplayed(self, udata):
        return weekly_prop_replayed(udata).std()
    
    def avgWeeklyPropInterrupted(self, udata):
        return weekly_prop_interrupted(udata).mean()

    def stdWeeklyPropInterrupted(self, udata):
        return weekly_prop_interrupted(udata).std()

    def freqAllActions(self, udata):
        """Compute the frequency of actions performed per hour spent watching videos"""
        udata = udata.copy()
        udata.loc[:, 'Day'] = udata.loc[:, 'Date'].dt.date  # Create column with the date but not the time
        udata.drop_duplicates(subset=['VideoID', 'Day'], inplace=True)  # Only keep on event per video per day
        durations = get_dated_videos()
        udata = udata.merge(durations, on=["VideoID", "Year"])
        watching_time = udata.Duration.sum() / 3600  # hours
        return total_actions(udata) / watching_time if watching_time != 0 else 0
    
    def freqPlay(self, udata):
        return count_actions(udata, 'Video.Play') / total_actions(udata)

    def freqPause(self, udata):
        return count_actions(udata, 'Video.Pause') / total_actions(udata)

    def freqSeekBackward(self, udata):
        return count_actions(udata, 'Video.SeekBackward') / total_actions(udata)

    def freqSeekForward(self, udata):
        return count_actions(udata, 'Video.SeekForward') / total_actions(udata)

    def freqSpeedChange(self, udata):
        return count_actions(udata, 'Video.SpeedChange') / total_actions(udata)

    def freqStop(self, udata):
        return count_actions(udata, 'Video.Stop') / total_actions(udata)
    
    def avgPauseDuration(self, udata):
        return pause_duration(udata)[0].mean()

    def stdPauseDuration(self, udata):
        return pause_duration(udata)[0].std()
    
    def avgSeekLength(self, udata):
        return seek_length(udata).mean()

    def stdSeekLength(self, udata):
        return seek_length(udata).std()

    def avgTimeSpeeding_up(self, udata):
        return compute_time_speeding_up(udata).mean()

    def stdTimeSpeedingUp(self, udata):
        return compute_time_speeding_up(udata).std()