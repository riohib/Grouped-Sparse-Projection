
#============================ POST PROCESSING ================================

fig=plt.figure(0)
plt.plot(loss_array)
with open("./Loss/loss.txt", "wb") as fp:   #Pickling
    pickle.dump(loss_array, fp)

fig.savefig('./Loss/lossCurve.png', bbox_inches='tight', dpi=100)

# ========================== Log loss_array to file ===================================
def save_loss(loss_array):
    la=[]
    for l in loss_array:
        tmp = l.detach().numpy()
        la.append(tmp)
        np.savetxt("./Loss/loss_s.csv", la, delimitr=",")
save_loss(loss_array)


# ================================ Plot L1 ======================================
plt.figure(2)
weights = model.e1.weight.detach()
io = to_img(weights)
plt.imshow(io[20].view(28,28))
save_image(io, './weights_plot/wp.png'.format(epoch))


print( "Any negative number in L1 weights: " + str(True in list))
print("Total itrations: " + str(itr) )
print("Sparsity Applied: " + str(in_sparse) )
print("Any Nan: " + str(True in nan_list))
print("nan in itr: " + str(nan_itr))


sModel = Net()
sModel.load_state_dict(torch.load('LeNet300.pth', map_location='cpu'))
weight_splot(1, sModel)