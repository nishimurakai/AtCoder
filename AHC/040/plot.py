# def plot_shapes(x_list, y_list, filename):
#     fig, ax = plt.subplots()
#     for i in range(len(x_list)):
#         x0, x1 = x_list[i]
#         y0, y1 = y_list[i]
#         rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None, edgecolor='r')
#         ax.add_patch(rect)
#     ax.set_xlim(0, max(x[1] for x in x_list))
#     ax.set_ylim(0, max(y[1] for y in y_list))
#     ax.invert_yaxis()
#     ax.xaxis.tick_top()
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.savefig(filename)