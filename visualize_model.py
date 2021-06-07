import matplotlib.pyplot as plt
import torch

def visualize_model(model, num_images=6):
    was_traing = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(numpy//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted : {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_traing)
                    return
    model.train(mode=was_traing)
