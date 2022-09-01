#questo script serve a comparare divcersi modelli e a plottare le relative loss function di validation e test 
#per far andare questo script sono nevìcessari i file .npy che contengono il salvataggio dei pesi dei modleli usati 
#non è necessario far andare lo script, rimanendo un draft utile per lavori futuri

history_noDropout = np.load('my_history_single_eta_model_noDropout.npy',allow_pickle='TRUE').item()

history_Dropout0 = np.load('my_history_single_eta_model_Dropout0.npy',allow_pickle='TRUE').item()


history_1 = np.load('my_history_single_eta_model_2.npy',allow_pickle='TRUE').item()

history_SGD = np.load('my_history_single_eta_model_SGD.npy',allow_pickle='TRUE').item()

history_L2e_6 = np.load('my_history_single_eta_model_L2e-6.npy',allow_pickle='TRUE').item()
history_L2e_8 = np.load('my_history_single_eta_model_L2e-8.npy',allow_pickle='TRUE').item()

#plot di varie loss function di train 
#plt.plot( history["loss"], label = 'Dropout 0.2 ' , color = 'b')


#plt.plot( history_noDropout["loss"], label = 'Dropout 0.1 Adam',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_Dropout0["loss"], label = 'Dropout 0.0 Adam $L_{2}=10^{-7}$',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_1["loss"], label = 'Dropout 0.2 Adam $ L_{2}=10^{-7}$',color = 'orange' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_SGD["loss"], label = 'SGD optimizer',color = 'black' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test


plt.plot( history_L2e_6["loss"], label = 'Adam - $ L_{2}=10^{-6}$',linestyle='--',color = 'red' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.plot( history_L2e_8["loss"], label = 'Adam - $L_{2}=10^{-8}$',linestyle='--',color = 'cyan' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.yscale("log")
plt.ylabel("training loss function")
plt.xlabel("epochs ")
plt.legend()
plt.grid(color='k', linestyle='--', linewidth=0.1)
plt.show()

#plot di varie loss function di validation 
#plt.plot( history["val_loss"], label = 'Dropout 0.2 Adam' , color = 'b') 


#plt.plot( history_noDropout["val_loss"], label = 'Dropout 0.1 Adam Optimizer',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_Dropout0["val_loss"], label = 'Dropout 0.0 Adam $ L_{2}=10^{-7}$',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test


plt.plot( history_1["val_loss"], label = 'Dropout 0.2 - Adam $ L_{2}=10^{-7}$',color = 'orange' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_SGD["val_loss"], label = 'Dropout 0.2 - SGD optimizer',color = 'black' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test


plt.plot( history_L2e_6["val_loss"], label = 'Adam - $ L_{2}=10^{-6}$',linestyle='--',color = 'red' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.plot( history_L2e_8["val_loss"], label = 'Adam - $L_{2}=10^{-8}$',linestyle='--',color = 'cyan' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.yscale("log")
plt.ylabel("validation loss function ")
plt.xlabel("epochs ")
plt.legend()
plt.grid(color='k', linestyle='--', linewidth=0.1)
plt.show()


