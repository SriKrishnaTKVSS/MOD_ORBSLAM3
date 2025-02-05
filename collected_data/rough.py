# # import numpy as np
# # import matplotlib.pyplot as plt

# # T_f=2*np.pi
# # N=500
# # t=np.linspace(0,T_f,N)

# # dt=T_f/N

# # x=np.sin(t)

# # integ_x_analytical=1-np.cos(t)

# # numerical_integ_x=np.zeros_like(x)
# # numerical_integ_x[0]=x[0]
# # for i in range(0,len(t)-1):
# #     numerical_integ_x[i+1]=dt*np.trapz(x[:i])+x[0]


# # # ------- simulating observer on synthetic data ------------

# # # plots
# # plt.figure(1)
# # plt.plot(t,integ_x_analytical,label='analytical')
# # plt.plot(t,numerical_integ_x,label='trapz')
# # plt.legend()
# # plt.xlabel('t(s)')
# # plt.ylabel('signal')
# # plt.grid(True)
# # plt.title('Integration comparison')
# # plt.show()


# ###----------------- Plotting with parameters ----------------
# import matplotlib.pyplot as plt

# # Adjust font sizes globally
# plt.rcParams.update({
#     'font.size': 14,          # Increase font size
#     'axes.titlesize': 16,     # Title font size
#     'axes.labelsize': 14,     # Axis labels font size
#     'legend.fontsize': 12,    # Legend font size
#     'xtick.labelsize': 12,    # X-axis tick labels font size
#     'ytick.labelsize': 12,    # Y-axis tick labels font size
#     'savefig.dpi': 300        # Set resolution for saved figures (DPI)
# })

# # Generate a plot
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 5, 7, 11]

# plt.plot(x, y, label="Sample Data")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Sample Plot")
# plt.legend()

# # Save as a high-resolution PDF
# plt.savefig("edited_plot.pdf", format="pdf")
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Example 2D list: each inner list represents a dataset
data = [
    [1, 2, 2, 3, 4],  # Dataset 1
    [2, 3, 3, 4, 5],  # Dataset 2
    [1, 1, 2, 3, 3]   # Dataset 3
]

# Define the number of bins
bins = np.linspace(0, 6, 7)  # Create bins from 0 to 6 with 7 edges

# Calculate histograms for each dataset
hist_data = [np.histogram(dataset, bins=bins)[0] for dataset in data]

# Grouped bar chart setup
x = np.arange(len(bins) - 1)  # X positions for bins
width = 0.3  # Width of each bar
offsets = [-width, 0, width]  # Offset for each dataset bar

# Plot the data
fig, ax = plt.subplots(figsize=(8, 6))
for i, (hist, offset) in enumerate(zip(hist_data, offsets)):
    ax.bar(x + offset, hist, width=width, label=f"Dataset {i+1}")

# Customize the plot
ax.set_xticks(x)
ax.set_xticklabels([f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)])
ax.set_xlabel("Value Range")
ax.set_ylabel("Frequency")
ax.set_title("Grouped Histogram")
ax.legend()

plt.tight_layout()
plt.show()

