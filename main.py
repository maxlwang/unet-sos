from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


data_gen_args2 = dict()

#train_path = './data2/train'
#test_path = './data2/test'

train_path = '/sim/asinghvi/wheat_processed_v2/train/'
test_path = '/sim/asinghvi/wheat_processed_v2/test/'

myGene = trainGenerator(8,train_path,'image','label',data_gen_args2,save_to_dir = None)

model = unet()
#model = unet('unet_sos.hdf5')
model_checkpoint = ModelCheckpoint('unet_sos.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=250,epochs=20,callbacks=[model_checkpoint])
#model.fit_generator(myGene,epochs=1,callbacks=[model_checkpoint])


testGene = testGenerator(test_path, 100)
results = model.predict_generator(testGene,100,verbose=1)
saveResult(test_path,results)
