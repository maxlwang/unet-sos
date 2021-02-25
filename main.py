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
myGene = trainGenerator(5,'data2/train','image','label',data_gen_args2,save_to_dir = None)

model = unet()
#model = unet('unet_sos.hdf5')
model_checkpoint = ModelCheckpoint('unet_sos.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data2/test")
results = model.predict_generator(testGene,10,verbose=1)
saveResult("data2/test",results)
