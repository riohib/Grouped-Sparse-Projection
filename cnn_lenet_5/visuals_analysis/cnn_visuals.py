subRow = 4
subCol = 5
c = 0
plt.figure(figsize=(15,10))
fig, axes = plt.subplots(subRow, subCol)
plt.subplots_adjust(wspace=0.5, hspace=0.5)


for i in range(20):
    plt.subplot(subRow, subCol, c + 1)
    im = plt.imshow(model_dh.conv1.weight[i].detach().view(5,5), cmap=plt.cm.RdBu_r)
    c+=1

plt.set_title("DeepHoyer Conv Layer 1 Filters")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)