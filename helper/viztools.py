import matplotlib.pyplot as plt
import seaborn as sns

positions = ["top", "left", "right"]


def turn_spines_off(ax, positions=positions):
    for i in positions:
        ax.spines[i].set_visible(False)


def count_plots(x_col, y_col, data, tick_labels, padding, label_color='white', hue_col=None, title=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=x_col, data=data, ax=ax, hue=hue_col)
    plt.suptitle(title, fontsize=15)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([])
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_xlabel(x_col, fontsize=12)
    turn_spines_off(ax)
    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i], padding=padding, fontsize=14, color=label_color)
    plt.show()


def dist_plots(x_col, data, bins=15):
    fig, ax = plt.subplots(figsize=(8, 5))
    turn_spines_off(ax)
    sns.histplot(x=x_col, data=data, ax=ax, bins=bins)
    plt.axvline(x=data[x_col].mean(), color="red")
    plt.title(f"Distribution of {x_col}", fontsize=15)
    plt.ylabel("Number of Individuals", fontsize=12)
    plt.xlabel(x_col, fontsize=8)
    plt.show()


def bar_compare(data, x, y, tick_labels, title, hue=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue is None:
        sns.barplot(x=x, y=y, data=data, ax=ax)
    else:
        sns.barplot(x=x, y=y, data=data, ax=ax, hue=hue)
    plt.suptitle(title, fontsize=15)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([])
    ax.set_ylabel(y, fontsize=12)
    ax.set_xlabel(x, fontsize=12)
    turn_spines_off(ax)
    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i], padding=-30, fontsize=14, color="white")

def percent_plot(x, data, hue=None, title='', rotate_labels=0):
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue is None:
        sns.histplot(
        x=x,
        data=data,
        ax=ax,
        stat="percent",
        shrink=0.9,
        multiple="dodge",
        common_norm=False,
    )
    else:
        sns.histplot(
        x=x,
        data=data,
        ax=ax,
        hue=hue,
        stat="percent",
        shrink=0.9,
        multiple="dodge",
        common_norm=False,
    )
    plt.suptitle(title, fontsize=15)
    turn_spines_off(ax)
    ax.set_ylabel('Percent', fontsize=12)
    ax.set_xlabel(x, fontsize=12)
    ax.set_yticks([])
    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i], rotation=rotate_labels, fmt='%.2f', padding=5)