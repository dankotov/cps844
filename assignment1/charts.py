
# labels = list(classifiers.keys())
# accuracy_scores = []
# time_scores = []
# for label in labels:
#     accuracy_scores.append(classifiers[label]["accuracy"])
#     time_scores.append(classifiers[label]["runtime"])



# plt.figure(figsize=(12,8))
# style.use('ggplot')

# ax = pd.Series(accuracy_scores).plot(kind='barh', color='#08375f')
# ax.set_title('Accuracy Scores of Classifiers')
# ax.set_yticklabels([])
# ax.set_xlabel('Accuracy')
# ax.set_ylabel('Classifiers')

# rects = ax.patches
# counter = 0
# for rect in rects:
#     x_value = rect.get_width()
#     y_value = rect.get_y() + rect.get_height() / 2

#     label = labels[counter]

#     plt.annotate(
#         label,
#         (0.01, y_value),
#         va='center', 
#         color='white'
#     )
#     counter += 1

# plt.savefig('all_acc.png')

# plt.figure(figsize=(12,8))
# style.use('ggplot')

# ax = pd.Series(time_scores).plot(kind='barh', color='#08375f')
# ax.set_title('Time Scores of Classifiers')
# ax.set_yticklabels([])
# ax.set_xlabel('Accuracy')
# ax.set_ylabel('Classifiers')

# rects = ax.patches
# counter = 0
# for rect in rects:
#     x_value = rect.get_width()
#     y_value = rect.get_y() + rect.get_height() / 2

#     label = labels[counter]

#     if label == 'SV (kernel=linear)':
#         plt.annotate(
#             label,
#             (12, y_value),
#             va='center', 
#             color='white'
#         )
#     else:
#         plt.annotate(
#             label,
#             (x_value + 1, y_value),
#             va='center', 
#             color='black'
#         )
#     counter += 1

# plt.savefig('all_time.png')