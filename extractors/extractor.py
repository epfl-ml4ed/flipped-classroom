from helpers.feature_extraction import *

class Extractor():

    def __init__(self, name='base'):
        self.name = name

    def getName(self):
        return self.name

    #### Lemay Doleck ####
    def totalEvents(self, udata, event_type, unique=False):
        """
        Counts the number of events of the specified type.
        Columns required: EventType, VideoID
        """
        edata = udata[udata['EventType'] == event_type]
        return len(edata) if not unique else len(edata.drop_duplicates(subset=['VideoID'], keep='first'))
    
    ### AIED Lalle Conati ###
    def totalActions(self, udata):
        """Counts the total number of actions performed across every videos"""
        return len(udata)

    def totalViews(self, udata):
        """ 
        Counts the total of videos views (rewatch and interruption included)
        Assumption: consider that a video is watched at most once per day
        Columns required: VideoID, Date (datetime object)
        """
        copy = udata.copy()
        copy['Day'] = udata.Date.dt.date
        # From the assumption the video view is a unique pair (video id, day)
        return len(copy.drop_duplicates(subset=['VideoID', 'Day']))
    
    def avgWeeklyPropWatched(self, udata):
        """
        Computess the average proportion of videos watched per week. In other
        words the number of videos watched (only counting the ones assigned) over
        the total number of videos assigned.
        Columns required: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_watched(udata).mean()

    def stdWeeklyPropWatched(self, udata):
        """
        Computess the standard deviation of the proportions of videos watched over the weeks. 
        In other words the number of videos watched (only counting the ones assigned) 
        over the total number of videos assigned.
        Columns required: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_watched(udata).std()
    
    def avgWeeklyPropReplayed(self, udata):
        """
        Computess the average proportion of videos replayed per week. That is, for 
        each week  (nb of videos replayed / nb of videos assigned).
        Columns required: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_replayed(udata).mean()

    def stdWeeklyPropReplayed(self, udata):
        """
        Computess the standard deviation of the proportion of videos replayed over the weeks. 
        Columns required: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_replayed(udata).std()
    
    def avgWeeklyPropInterrupted(self, udata):
        """
        Computess the average proportion of videos interrupted per week. 
        A video is considered interrupted when
            * a break is too long
            * a break (not in the last minute) is followed by an event 
              in another video (the user left the video)
            * an event occurs in another video before the end of the current video
        Columns required: VideoID, Year (YYYY format), Date (datetime object), EventType,
                          TimeStamp, Duration, 
        """
        return weekly_prop_interrupted(udata).mean()

    def stdWeeklyPropInterrupted(self, udata):
        """
        Computess the standard deviation of the proportions of videos interrupted over the weeks. 
        Columns required: VideoID, Year (YYYY format), Date (datetime object), EventType,
                          TimeStamp, Duration, 
        """
        return weekly_prop_interrupted(udata).std()

    def freqAllActions(self, udata):
        """
        Computess the frequency of actions performed per hour spent watching videos
        Columns required: VideoID, Date (datetime object), Duration
        """
        udata = udata.copy()
        udata.loc[:, 'Day'] = udata.loc[:, 'Date'].dt.date  # Create column with the date but not the time
        udata.drop_duplicates(subset=['VideoID', 'Day'], inplace=True)  # Only keep on event per video per day
        watching_time = udata.Duration.sum() / 3600  # hours
        return total_actions(udata) / watching_time if watching_time != 0 else 0
    
    def freqPlay(self, udata):
        """
        Computess the ratio of Play events over the total number of actions
        Columns required: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Play') / total_actions(udata)

    def freqPause(self, udata):
        """
        Computess the ratio of Pause events over the total number of actions
        Columns required: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Pause') / total_actions(udata)

    def freqSeekBackward(self, udata):
        """
        Computess the ratio of Seek Backward events over the total number of actions
        Columns required: VideoID, Date (datetime object), EventType, OldTime,  NewTime
        """
        return count_actions(udata, 'Video.SeekBackward') / total_actions(udata)

    def freqSeekForward(self, udata):
        """
        Computes the ratio of Seek Forward events over the total number of actions
        Columns required: VideoID, Date (datetime object), EventType, OldTime,  NewTime
        """
        return count_actions(udata, 'Video.SeekForward') / total_actions(udata)

    def freqSpeedChange(self, udata):
        """
        Computes the ratio of SpeedChange events over the total number of actions
        Columns required: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.SpeedChange') / total_actions(udata)

    def freqStop(self, udata):
        """
        Computes the ratio of Stop events over the total number of actions
        Columns required: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Stop') / total_actions(udata)
    
    def avgPauseDuration(self, udata):
        """
        Computes the average time interval between each pause event and the next play event
        Only pause durations smaller than ~8 min are taken into account. 
        Columns required: EventType, TimeStamp
        """
        return pause_duration(udata).mean()

    def stdPauseDuration(self, udata):
        """
        Computes the standard deviation of the time intervals between each pause event and 
        the next play event. Only pause durations smaller than ~8 min are taken into account. 
        Columns required: EventType, TimeStamp
        """
        return pause_duration(udata).std()
    
    def avgSeekLength(self, udata):
        """
        Computes the average seek length. In other words, how much time is skipped
        forward/backward.
        Columns required: EventType, OldTime, NewTime
        """
        return seek_length(udata).mean()

    def stdSeekLength(self, udata):
        """
        Computes the standard deviation of seek lengths.
        Columns required: EventType, OldTime, NewTime
        """
        return seek_length(udata).std() 

    def avgTimeSpeeding_up(self, udata):
        """
        Computes the average time spent at a speed higher than 1 per video.
        Columns required: VideoID, Timestamp, EventType, CurrentTime, Duration
        """
        return compute_time_speeding_up(udata).mean()

    def stdTimeSpeedingUp(self, udata):
        """
        Computes the standard deviation of the time spent at a speed higher than 1 per video.
        Columns required: VideoID, Timestamp, EventType, CurrentTime, Duration
        """
        return compute_time_speeding_up(udata).std()

 
    ### Regularity Bouroujeni et al ###
  
    def peakDayHour(self, udata):
        """
        Identifies if user’s activities are concentrated around a 
        particular hour of the day
        Column required: TimeStamp
        """
        return PDH(udata)

    def peakWeekDay(self, udata):
        """
        Identifies if user’s activities are concentrated around a 
        particular day of the week
        Column required: TimeStamp
        """
        return PWD(udata)

    def weeklySimilarity1(self, udata):
        """
        Measures if the user works on the same weekdays 
        throughout the weeks.
        Columns required: TimeStamp
        """
        return WS1(udata)

    def weeklySimilarity2(self, udata):
        """
        Compares the normalized profiles and measure if the user has a similar 
        distribution of workload among weekdays, in different weeks of the course.
        Columns required: TimeStamp
        """
        return WS2(udata)

    def weeklySimilarity3(self, udata):
        """
        Compares the original profiles and reflects if the time spent on
        each day of the week is similar for different weeks of the course.
        Columns required: TimeStamp
        """
        return WS3(udata)

    def freqDayHour(self, udata):
        """
        Evaluates the intensity of a daily period, i.e., if the user
        works periodically at a specific hour of the day.
        The value is the Fourier transform of the active days (day with at least one event) 
        evaluated at the frequency 1 / nb of hours in a day = 1 / 24
        Columns required: TimeStamp
        """
        return FDH(udata)

    def freqWeekDay(self, udata):
        """
        Evaluates the intensity of a weekly period, i.e., if the user
        works periodically on a specific day of the week.
        The value is the Fourier transform of the active days (day with at least one event) 
        evaluated at the frequency 1 / nb of days in a week = 1 / 7
        Columns required: TimeStamp
        """
        return FWD(udata)

    def freqWeekHour(self, udata):
        """
        Evaluates the intensity of the period 7*24, i.e., if the user
        works periodically at a specific hour of the week.
        The value is the Fourier transform of the active days (day with at least one event) 
        evaluated at the frequency 1 / nb of hours in a week = 1 / (7*24)
        Columns required: TimeStamp
        """
        return FWH(udata)

    def nbQuiz(self, udata):
        """
        Total count of quiz completed over the whole semester
        Columns required: ProblemID
        """
        return NQZ(udata)

    def propQuiz(self, udata):
        """
        Proportion of quiz completed over the flipped period
        Columns required: Year
        """
        return PQZ(udata)
 
    def intervalVideoQuiz(self, video_events, problem_events):
        """
        For every completed quiz, compute the time intervals (minutes)
        between the first viewing of the video and the quiz completion
        and return the interquartile range of the time intervals
        video_events columns required: VideoID, EventType, TimeStamp, Subchapter
        problem_events columns required: ProblemID, EventType, TimeStamp, Subchapter
        """
        return IVQ(video_events, problem_events)

    def semesterRepartitionQuiz(self, udata):
        """
        Measures the repartition of the quiz completions. The std (in hours) of the time intervals is computed
        aswell as the dates of completions. The smaller the std is, the more regular the student is.
        Columns required: AccountUserID, ProblemID, EventType, TimeStamp
        """
        return SRQ(udata)