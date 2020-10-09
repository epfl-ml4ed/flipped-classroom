from helpers.data_process import getStudentTimeStamps
from helpers.feature_extraction import *
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hourly_activity(X, studentID=0):
    sid, T, Lw = getStudentTimeStamps(X, studentID)
    PDHs = "{:.2f}".format(PDH(Lw, T))
    plt.bar(np.arange(1, 25), dailyActivity(Lw * 7, T))
    plt.title("Student " + sid + " Activity per Hour, PDH = " + PDHs)

def plot_weekly_activity(X, studentID=0):
    sid, T, Lw = getStudentTimeStamps(X, studentID)
    PWDs = "{:.2f}".format(PWD(Lw, T))
    plt.bar(np.arange(1, 8), weeklyActivity(Lw, T))
    plt.ylim(0, 10)
    plt.title("Student " + sid + " Activity per Week, PWD = " + PWDs)

def plot_WS(X, studentID=0):
    sid, T, Lw = getStudentTimeStamps(X, studentID)
    WS = "{:.2f}, {:.2f}, {:.2f}".format(WS1(Lw, T), WS2(Lw, T), WS3(Lw, T))
    ax = sns.heatmap(dayActivityByWeek(Lw, T), cmap=sns.cubehelix_palette(as_cmap=True))
    ax.invert_yaxis()
    plt.title("Weekly similarities with (WS1, WS2, WS3) = (" + WS+")")
    plt.ylabel('Week')
    plt.xlabel('Day')

def plot_Fourier(X, studentID=0):
    sid, T, Lw = getStudentTimeStamps(X, studentID)
    FWD_value = FWD(Lw, T)
    fourier_values = "{:.2f}, {:.2f}, {:.2f}".format(FDH(Lw, T), FWH(Lw, T), FWD_value)
    print("FDH, FWH, FWD = " + fourier_values)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    activity = np.array([studentActivity(24*60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(24*60*60))])
    days = np.arange(len(activity))
    line1 = sns.lineplot(ax=axes[0], x=np.arange(len(activity)), y=activity)
    for week in range(7, days[-1], 7):
        if week == 7:
            axes[0].axvline(x=week, c='gray', ls='--', label="Weeks")
        else:
            axes[0].axvline(x=week, c='gray', ls='--')
    line1.set_title("Student activity over the days")
    line1.set_ylabel('Active or not')
    line1.set_xlabel('Day')
    axes[0].legend()

    freq = np.arange(0.1, 0.5, 1 / 70)
    spectrum = [FWD(Lw, T, f) for f in freq]
    line2 = sns.lineplot(ax=axes[1], x=freq, y=spectrum)
    axes[1].axvline(x=1 / 7, label="1 / 7", c='red', ls='--')
    line2.set_ylim(0, 15)
    line2.set_title("Spectrum with FWD = {:.2f}".format(FWD_value))
    line2.set_ylabel('Spectrum')
    line2.set_xlabel('Frequency')
    axes[1].legend()

def plotSilhouette(sse):
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, len(sse) + 2), sse)
    plt.xticks(range(len(sse) + 2))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.xlim([0, len(sse)+2])
    plt.ylim([0, 0.5])
    plt.show()