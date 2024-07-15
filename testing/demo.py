import torch
import matplotlib.pyplot as plt

from utils import define_device
import dataPreparing.loaders as loaders


# Define computation device and load models.
device = define_device()
models_loader = loaders.ModelsLoader(device)

_, preprocess = models_loader.load_vit_b_16()



# model = models_loader.load_local("./models/gc-simple-dnn-1.pth")
model = models_loader.load_local("./models/gc-simple-dnn-2.pth")

# Load test data.
datasets_loader = loaders.DatasetsLoader(device, preprocess)
test_data = datasets_loader.load_test_celeba_images(root="../data")

# Label descriptions of result scalars.
labels_map = {
    0: "Female",
    1: "Male",
}


def random_images_prediction(data, length=3):
    figure = plt.figure(figsize=(8, 10))

    for i in range(length ** 2):
        sample_idx = torch.randint(len(data), size=(1,)).item()

        img, answer = data[sample_idx]
        answer = int(answer.item())

        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(x).item()

        figure.add_subplot(length, length, i + 1)
        gender_class = round(predict)

        if gender_class == 0:
            prob = 1.0 - predict
        else:
            prob = predict

        title = (f"Predict: {labels_map[gender_class]} with {prob:0.2f}\n"
                 f"Answer: {labels_map[answer]}")

        if gender_class == answer:
            color = "green"
        else:
            color = "red"

        plt.title(label=title, color=color)
        plt.axis("off")
        plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    random_images_prediction(data=test_data)
