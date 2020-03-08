import TensorFlowModel
import tkinter as tk
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory
import _thread


def classification():
    answer = model.core_recognize(app.path)
    prediction_label = 'Prediction: '
    app.lbl_answer.config(text=prediction_label)
    if answer > 0.5:
        app.lbl_answer.config(text=prediction_label + 'DOG')
    else:
        app.lbl_answer.config(text=prediction_label + 'CAT')


def summary():
    messagebox.showinfo('Summary', model.core_summary())


def create():
    messagebox.showinfo('TensorFlow Model', 'Empty model created')
    app.modelcreator.lbl_model_status.config(text='Model ready to work')
    model.core_create()


def compile():
    model.core_compile()
    app.modelcreator.hide()


def build():
    model.core_build()


def dense():
    n = int(app.modelcreator.txt_model_dense.get())
    model.core_add_dense(n)
    app.modelcreator.txt_model_dense.config(text='')


def flattern():
    model.core_add_flattern()


def conv2d():
    n = int(app.modelcreator.txt_model_conv2d.get())
    model.core_add_conv2d(n)
    app.modelcreator.txt_model_conv2d.config(text='')


def maxpool():
    model.core_add_maxpooling2()


def dropout():
    model.core_add_dropout()


def training():
    path = askdirectory()
    model.core_training_set(path)
    messagebox.showinfo("Training", "Training set selected")


def validation():
    path = askdirectory()
    model.core_validation_set(path)
    messagebox.showinfo("Validation", "Validation set selected")


def fit():
    _thread.start_new_thread(model.core_train, ())


class ModelCreator:
    def __init__(self, master):
        self.window_model = tk.Toplevel(master)

        self.lblfr_model = tk.LabelFrame(self.window_model, text='Model')
        self.btn_model_create = tk.Button(self.lblfr_model, text='Create new Model', command=create)

        self.lbl_model_view = tk.Label(self.lblfr_model, text='Add layer:')

        self.btn_model_dense = tk.Button(self.lblfr_model, text='Add', command=dense)
        self.lbl_model_dense = tk.Label(self.lblfr_model, text='Dense')
        self.txt_model_dense = tk.Entry(self.lblfr_model, width='7')
        self.txt_model_dense.grid(column='1', row='2')

        self.btn_model_flattern = tk.Button(self.lblfr_model, text='Add', command=flattern)
        self.lbl_model_flattern = tk.Label(self.lblfr_model, text='Flattern')

        self.btn_model_maxpool = tk.Button(self.lblfr_model, text='Add', command=maxpool)
        self.lbl_model_maxpool = tk.Label(self.lblfr_model, text='MaxPooling2D')
        self.lbl_model_maxpool.grid(column='0', row='6')
        self.btn_model_maxpool.grid(column='1', row='6')

        self.btn_model_conv2d = tk.Button(self.lblfr_model, text='Add', command=conv2d)
        self.lbl_model_conv2d = tk.Label(self.lblfr_model, text='Conv2D')
        self.txt_model_conv2d = tk.Entry(self.lblfr_model, width='7')
        self.txt_model_conv2d.grid(column='1', row='4')

        self.btn_model_sum = tk.Button(self.lblfr_model, text='Compile', command=compile)

        self.btn_model_dropout = tk.Button(self.lblfr_model, text='Add', command=dropout)
        self.lbl_model_dropout = tk.Label(self.lblfr_model, text='Dropout')

        self.lbl_model_status = tk.Label(self.lblfr_model, text='New model is not ready')

        self.btn_model_create.grid(column='0', row='0')
        self.lbl_model_status.grid(column='1', row='0')
        self.lblfr_model.grid(column='1', row='1')
        self.lbl_model_view.grid(column='0', row='1')
        self.btn_model_sum.grid(column='6', row='6')
        self.lbl_model_dense.grid(column='0', row='2')
        self.lbl_model_flattern.grid(column='0', row='3')
        self.lbl_model_conv2d.grid(column='0', row='4')
        self.lbl_model_dropout.grid(column='0', row='5')
        self.btn_model_dense.grid(column='2', row='2')
        self.btn_model_flattern.grid(column='1', row='3')
        self.btn_model_conv2d.grid(column='2', row='4')
        self.btn_model_dropout.grid(column='1', row='5')

    def hide(self):
        self.window_model.withdraw()

    def show(self):
            self.window_model.deiconify()


class GUI:
    def openfile(self):
        new_path = askopenfilename(parent=self.root, initialdir='/', filetypes=[('allfiles', '*.*')])
        self.lbl_answer.configure(text="Classify to get prediction")
        if new_path is not None:
            self.path = new_path
            self.image = ImageTk.PhotoImage(Image.open(self.path).resize((300, 300)))
            self.lbl_image.config(image=self.image)
            self.lbl_image.image = self.image

    def model_save(self):
        path = asksaveasfilename(parent=self.root, filetypes=[('Models', '*.h5')], title='Save Model')
        model.core_save(path+'.h5')

    def model_load(self):
        path = askopenfilename(parent=self.root, initialdir='/', filetypes=[('allfiles', '*.h5')])
        model.core_load(path)
        messagebox.showinfo('Model Load', 'Loaded')

    def model_create(self):
        if self.modelcreator is not None:
            self.modelcreator.show()
        else:
            self.modelcreator = ModelCreator()

    def model_directory(self):
        pass

    def __init__(self, root):
        self.root = root
        root.title("Neural Network v0.3")
        self.menu = tk.Menu(root)
        self.root.config(menu=self.menu)
        self.modelcreator = ModelCreator(root)
        self.modelcreator.hide()

        self.menu_file = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=self.menu_file)
        self.menu_file.add_command(label="Open image", command=self.openfile)
        self.menu_file.add_command(label="Exit", command=self.root.quit)

        self.menu_model = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="Model", menu=self.menu_model)
        self.menu_model.add_command(label='Load', command=self.model_load)
        self.menu_model.add_command(label='Save', command=self.model_save)
        self.menu_model.add_command(label='Create', command=self.model_create)
        self.menu_model.add_command(label='Summary', command=summary)

        self.menu_train = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label='Train', menu=self.menu_train)
        self.menu_train.add_command(label='Select train set', command=training)
        self.menu_train.add_command(label='Select validation set', command=validation)
        self.menu_train.add_command(label='Fit model', command=fit)

        self.lbl_answer = tk.Label(text='Classify to get prediction')
        self.lbl_answer.grid(column='0', row='1')

        self.lbl_image = tk.Label(self.root, text='Image')
        self.path = "template.jpg"
        if self.path is not None:
            self.image = ImageTk.PhotoImage(Image.open(self.path).resize((300, 300)))
            self.lbl_image = tk.Label(self.root, text='Image', image=self.image)
            self.lbl_image.image = self.image
            self.lbl_image.grid(column='0', row='0')

        self.btn_get = tk.Button(self.root, text='CLASSIFY', highlightcolor="#555", activebackground="#ffff00", command=classification)
        self.btn_get.grid(column='1', row='1')


root = tk.Tk()
app = GUI(root)
model = TensorFlowModel.Model()
root.mainloop()