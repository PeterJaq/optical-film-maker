import numpy as np 
import pandas as pd 

def film_loss(aim, weight, observation, average=False):
    # Calculate film loss 
    
    loss_absorbation   = np.mean(weight['Absorption'] * (abs(aim['Absorption'] - observation[0])))
    loss_transimission = np.mean(weight['Transmission'] * (abs(aim['Transmission'] - observation[1])))
    loss_refraction    = np.mean(weight['Refraction'] * (abs(aim['Refraction'] - observation[2])))

    if average:
        #print(np.sum([loss_absorbation, loss_transimission, loss_refraction]))
        return np.sum([loss_absorbation, loss_transimission, loss_refraction])
    else:
        return loss_absorbation, loss_transimission, loss_refraction

