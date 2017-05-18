


# Just to keep the lines in memory, but it does not work. 
def temp_visualization(self):
IDX_CLU = np.argsort(self.labels)

similarity_matrix_reordered = sim[IDX_CLU,:][:,IDX_CLU]

plt.imshow(similarity_matrix_reordered, interpolation='none')
plt.show()
plt.imshow(ROI_clu_sort, aspect='auto', interpolation='none');
plt.show()
plt.imshow(sim_mat_clusters);
plt.show()
