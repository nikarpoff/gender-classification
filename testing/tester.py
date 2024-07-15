import time
import sys
import torch

from torch import nn
from torch.utils.data import DataLoader

from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from utils import define_device
import dataPreparing.loaders as loaders


def test_model(testing_model, dataloader, base_data_path, device, batch_size, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    wrong_answers = []
    map_labels = {
        0: "Female",
        1: "Male"
    }

    print(f"Size of test dataset -> {size}\n"
          f"  Batch size -> {batch_size}\n"
          f"  Number of batches -> {num_batches}")

    test_loss, correct = 0., 0

    print("\nStart evaluation...")

    test_start = time.time()

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Prepare inputs.
            X, target = X.to(device), y[0][:].to(device)

            # Generate prediction.
            pred = testing_model(X.squeeze())
            prob = pred

            # Count metrics.
            test_loss += loss_function(pred, target).item()
            pred = torch.round(pred.squeeze())

            for i in range(pred.shape[0]):
                answer = y[0][i].item()

                # If it is right prediction then increase correct
                if pred[i] == target[i][0]:
                    correct += 1
                else:
                    # Remember answer and prediction.
                    text = (f"Wrong answer for image {y[1][i]}. "
                            f"Expected: {map_labels[answer]}. "
                            f"Predicted: {map_labels[pred[i].item()]} with {prob[i].item():0.2f}")

                    print(f'{text}\n')

                    wrong_answers.append((f'{base_data_path}/{y[1][i]}', text))

    test_loss /= num_batches
    accuracy = 100 * correct / size

    required_time = time.time() - test_start

    print(f"\nTest Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}, "
          f"Required time: {required_time:0.3f} s\n")

    return accuracy, test_loss, wrong_answers, required_time


def generate_pdf_report(dataset_name, loss, accuracy, wrong_answers, test_time, model_name):
    print("Start generate pdf report")

    # Get current datetime and use it for filename.
    current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M")
    filename = f"./reports/test-{current_datetime}.pdf"

    # Create PDF document.
    c = canvas.Canvas(filename, pagesize=letter)

    # Fill document with content
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Report with test {current_datetime} results for {model_name}")
    c.drawString(100, 730, f"Test dataset: {dataset_name}")
    c.drawString(100, 710, f"Average loss: {loss:0.3f}")
    c.drawString(100, 690, f"Accuracy: {accuracy}")
    c.drawString(100, 670, f"Required time: {test_time:0.3f} s")
    c.drawString(100, 650, f"Wrong answers {len(wrong_answers)}:")
    y_position = 630
    image_width = 160
    image_height = 180

    # Display all wrong answers with images.
    for answer in wrong_answers:
        # Check for page end
        if y_position - image_height < 90:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750

        y_position -= image_height

        # Draw wrong classified image
        c.drawImage(answer[0], 100, y_position, image_width, image_height)

        y_position -= 20
        c.drawString(100, y_position, answer[1])
        y_position -= 20

    # Save file.
    c.save()

    print(f"Report completed: {filename}")


if __name__ == "__main__":
    model_name = "gc-simple-dnn-2"
    dataset_name = "cafe"

    # First argv is model name, second - dataset name.
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        model_name = sys.argv[1]
    if len(sys.argv) == 3:
        dataset_name = sys.argv[2]

    # Define computation device and load models.
    device = define_device()
    models_loader = loaders.ModelsLoader(device)

    _, preprocess = models_loader.load_vit_b_16()

    model = models_loader.load_local(f"./models/{model_name}.pth")

    batch_size = 64

    # Load test data.
    datasets_loader = loaders.DatasetsLoader(device, preprocess)

    # Custom or CelebA.
    if dataset_name == 'custom':
        print("Use custom dataset")
        test_data = datasets_loader.load_custom_dataset()
        base_data_path = "./data/faces-dataset"
    elif dataset_name == 'cafe':
        print("Use cafe-video dataset")
        test_data = datasets_loader.load_cafe_dataset()
        base_data_path = "./data/cafe-faces"
    else:
        print("Use CelebA dataset")
        test_data = datasets_loader.load_test_celeba_with_path(root="./data")
        base_data_path = "./data/celeba/img_align_celeba"

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Use binary cross entropy loss
    loss_function = nn.BCELoss()

    # Run test.
    accuracy, test_loss, wrong_answers, required_time = test_model(testing_model=model, dataloader=test_dataloader,
                                                                   base_data_path=base_data_path, device=device,
                                                                   batch_size=batch_size, loss_function=loss_function)

    # Generate report.
    generate_pdf_report(dataset_name, test_loss, accuracy, wrong_answers, required_time, model_name)
