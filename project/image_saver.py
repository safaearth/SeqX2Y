import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# save the dvf.png for test
def save_dvf_image(DVF, batch_idx, savepath):
    dvf = DVF[0, :, 0, 0, ...]
    dvf = dvf.permute(1, 2, 0)
    dvf = dvf.cpu().detach().numpy()
    plt.imshow(dvf)
    plt.savefig(os.path.join(savepath, f"{batch_idx:03d}", "dvf.png"))

# save the Bat_pre.png for test
def save_bat_pred_image(bat_pred, batch_idx, savepath):
    Bat_Pred = bat_pred[0, 0, :, 0, ...]
    Bat_Pred = Bat_Pred.permute(1, 2, 0)
    Bat_Pred = Bat_Pred.cpu().detach().numpy()
    plt.imshow(Bat_Pred)
    plt.savefig(os.path.join(savepath, f"{batch_idx:03d}", "Bat_Pred.png"))

# save the inhale_predict.nrrd
def save_sitk_images(bat_pred, batch_idx, savepath):
    BAT_PRED = bat_pred.cpu().detach().numpy()  # Convert to numpy array once and use it
    BAT_PRED = np.squeeze(BAT_PRED)
    writer = sitk.ImageFileWriter()
    for i in range(3):  # Assuming 3 phases as in your example
        img_array = np.squeeze(BAT_PRED[i, ...])
        writer.SetFileName(os.path.join(savepath, f"{batch_idx:03d}", f"inhale{i+1}_predict.nrrd"))
        writer.Execute(sitk.GetImageFromArray(img_array))

# save the dvf.nrrd
def save_sitk_DVF_images(DVF, batch_idx, savepath):
    # Permute DVF & Save DVF
    def dvf_(d):
        x = d[0,...]
        x = np.reshape(x, [1,118, 128, 128])
        y = d[1,...]
        y = np.reshape(y, [1,118, 128, 128])
        z = d[2,...]
        z = np.reshape(z, [1,118, 128, 128])
        out = np.concatenate([z,y,x],0)
        return out
    
    Dvf = DVF.cpu().detach().numpy()  # Convert to numpy array
    Dvf = np.squeeze(Dvf)  # Remove singleton dimensions
    
    for i in range(3):  # Assuming 3 phases as in your example
        DVF_img = dvf_(Dvf[:, i, ...])
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(savepath, f"{batch_idx:03d}", f"DVF{i+1}.nrrd"))
        writer.Execute(sitk.GetImageFromArray(np.transpose(DVF_img,[1,2,3,0])))


# ------------------------------------------------------------------------
        # # save DVF img
        # savepath = '/workspace/SeqX2Y_PyTorch/test/Imageresult'

        # # make dir 
        # save_path = savepath + "/" + "%3.3d" % batch_idx
        # if not os.path.exists(save_path):os.makedirs(save_path)

        # # save dvf img
        # dvf=DVF[0,:,0,0,...]
        # dvf=dvf.permute(1,2,0)
        # dvf=dvf.cpu().detach().numpy()
        # plt.imshow(dvf)
        # plt.show()
        # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/dvf.png')

        # # save bat pred
        # Bat_Pred=bat_pred[0,0,:,0,...]
        # Bat_Pred=Bat_Pred.permute(1,2,0)
        # Bat_Pred=Bat_Pred.cpu().detach().numpy()
        # # plt.imshow(Bat_Pred)
        # # plt.show()
        # plt.savefig('/workspace/SeqX2Y_PyTorch/test/Imageresult/Bat_Pred.png')

        # # save predict img
        # BAT_PRED = bat_pred.cpu().detach().numpy() # 1, 1, future_seq, 128, 128, 128
        # BAT_PRED = np.squeeze(BAT_PRED) # pred_feat, 128, 128, 128
        
        # writer = sitk.ImageFileWriter()
        # pI1, pI2, pI3 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]) #seq = 3   

        # pI1, pI2, pI3, pI4 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]), np.squeeze(BAT_PRED[3, ...]) #seq = 4  
        # pI1, pI2, pI3, pI4 = np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[3, ...]), np.squeeze(BAT_PRED[5, ...]), np.squeeze(BAT_PRED[7, ...]) # other seq = 4     
        # pI1, pI2, pI3, pI4, pI5, pI6 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]), np.squeeze(BAT_PRED[3, ...]), np.squeeze(BAT_PRED[4, ...]), np.squeeze(BAT_PRED[5, ...]) #seq = 6      
        # pI1, pI2, pI3, pI4, pI5, pI6, pI7, pI8 = np.squeeze(BAT_PRED[0, ...]), np.squeeze(BAT_PRED[1, ...]), np.squeeze(BAT_PRED[2, ...]), np.squeeze(BAT_PRED[3, ...]), np.squeeze(BAT_PRED[4, ...]), np.squeeze(BAT_PRED[5, ...]), np.squeeze(BAT_PRED[6, ...]), np.squeeze(BAT_PRED[6, ...]) #seq = 7
        
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale1_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI1))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale2_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI2))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale3_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI3))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale4_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI4))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale5_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI5))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale6_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI6))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale7_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI7))
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "inhale8_predict.nrrd")
        # writer.Execute(sitk.GetImageFromArray(pI8))
        
        # # Permute DVF & Save DVF
        # def dvf_(d):
        #     x = d[0,...]
        #     x = np.reshape(x, [1,118, 128, 128])
        #     y = d[1,...]
        #     y = np.reshape(y, [1,118, 128, 128])
        #     z = d[2,...]
        #     z = np.reshape(z, [1,118, 128, 128])
        #     out = np.concatenate([z,y,x],0)
        #     return out
        
        # Dvf = DVF.cpu().detach().numpy() # 1,3,9, 128, 128
        # Dvf = np.squeeze(Dvf) # 3, 9, 128, 128, 128
        # DVF2, DVF3, DVF4 = dvf_(Dvf[:,0,...]), dvf_(Dvf[:,1,...]), dvf_(Dvf[:,2,...])

        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF2.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF2), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF3.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF3), [1,2,3,0]))) # 3 1 2
        # writer.SetFileName(savepath + "/" + "%3.3d" % batch_idx + "/" + "DVF4.nrrd")
        # writer.Execute(sitk.GetImageFromArray(np.transpose(dvf_(DVF4), [1,2,3,0]))) # 3 1 2