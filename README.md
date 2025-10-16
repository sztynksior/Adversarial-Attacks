# Adversarial-Attacks
This small example project demonstrates how to **train a perturbation mask** used in an adversarial attack against a road-signs image classification model.

## About Adversarial Attacks
Image-recognition neural networks are vulnerable to adversarial attacks. It is possible to cause a trained model to misclassify an image as a chosen target class by adding a trained perturbation (a "mask") to the image. The perturbation is usually constrained so the changes are small and often hard for humans to notice, yet large enough to mislead the model.

## Perturbation mask training

The following code block presents learning step function used for perturbation mask training:
```Python
def do_step():
    with tf.GradientTape() as tape:
        adv_image = tf.clip_by_value(image + delta, 0, 255)
        prediction = model(adv_image, training=False)
        original_loss = lossFunct(tf.convert_to_tensor([real_class]), prediction)
        target_loss = lossFunct(tf.convert_to_tensor([target_class]), prediction)
        loss = target_loss - original_loss

    gradients = tape.gradient(loss, delta)
    optimizer.apply_gradients([(gradients, delta)])
    clipped_delta = tf.clip_by_value(delta, clip_value_min=-0.01, clip_value_max=0.01)
    delta.assign_add(clipped_delta)

    return loss, prediction
```

### Loss function interpretation

Both **target_loss** and **original_loss** are categorical cross entropy losses. Knowing that we can expand formula of the final loss:

$$Loss = -\sum_{i=1}^n{t_i\log{\hat y_i} - o_i\log{\hat y_i}}$$

where $n$ is a number of classes, $t_i$ is equal to 1 only if $i =\text{target class}$ and $0$ otherwise, $o_i$ is equal to 1 only if $i =\text{oryginal class}$ and $0$ otherwise and $\hat y_i$ is a probability output of the model for class $i$. By assuming that for target class $i=a$ and for oryginal class $i=b$ we can simplify the loss formula:

$$Loss = -\log{\hat y_a}+\log{\hat y_b}$$

Now it is clearly visible how the loss affects the training. It benefits the perturbation mask if probability of target class gets higher and probability of oryginal class gets lower.

To understand how the gradient is calculated it is enaugh to imagine that perturbation mask is just another layer of the model added before the oryginal input layer, and we will call it adversarial attak layer from now. Each neuron of the layer takes as an input value of one of oryginal image pixels and ah one connections with corresponding neuron of the oryginal input layer. Adversarial attak layer activation function looks as follows:

$$y=x_i+b_i$$

where $x_i$ is a value of 

### Calculalting the gradient

The following formula describes how gradient is applayed to the perturbarion mask at each learning step:

$$\forall t, 𝐩^{(t)}\leftarrow 𝐩^{(t-1)}-\min(-0.01, \max(\eta\nabla_{𝐩^{(t-1)}}-\log(y_t)+\log(y_o), 0.01))$$

where $t$ is a learning step, 𝐩 is a perturbation mask and $\eta$ is a learning rate parameter. 

## Results
### Model performence before the attack
<table align="center">
  <thead>
      <th colspan="3">Initial predictions</th>
  </thead>
  <tbody>
    <tr>
      <th>Stop</th>
      <th>Pedestrians</th>
      <th>Yield</th>
    </tr>
    <tr>
      <th><img width="150" height="137" alt="stop" src="https://github.com/user-attachments/assets/f658049a-2f70-48c8-a7b3-cf28b18aed46" /></th>
      <th><img width="150" height="137" alt="pedestrians" src="https://github.com/user-attachments/assets/604d489c-5f03-4563-a372-b6ee963ead59" /></th>
      <th><img width="150" height="137" alt="yield" src="https://github.com/user-attachments/assets/674607a2-05b3-43c9-9e92-6cde0f26ccf3" /></th>
    </tr>
  </tbody>
</table>

### Model performence after the attack
<table align="center">
  <thead>
    <tr>
      <th scope="col" colspan="4">Experiment 1</th>
    </tr>
    <tr>
      <th scope="col" colspan="2">Oryginal image</th>
      <th scope="col" colspan="2">Target image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">name: Stop</th>
      <th scope="row">label: 14</th>
      <th scope="row">name: Pedestrians</th>
      <th scope="row">label: 27</th>
    </tr>
    <tr>
      <th scope="row" colspan="4"><img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/51773000-b34b-45f9-a4bf-303f73b3e359" />
    </tr>
    <tr>
      <th scope="row" colspan="2">Target class reached at atep 158</th>
      <th scope="row" colspan="2">Prediction: Pedestrians</th>
    </tr>
    <tr>
      <th scope="row" colspan="2">Loss: -23.124422073364258</th>
      <th scope="row" colspan="2">Target Class Probability: 0.40884917974472046</th>
    </tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr>
      <th scope="col" colspan="4">Experiment 2</th>
    </tr>
    <tr>
      <th scope="col" colspan="2">Oryginal image</th>
      <th scope="col" colspan="2">Target image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">name: Pedestrians</th>
      <th scope="row">label: 27</th>
      <th scope="row">name: Yield</th>
      <th scope="row">label: 13</th>
    </tr>
    <tr>
      <th scope="row" colspan="4"><img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/2bdb744d-9e5e-4141-93eb-daebeb1575da" />
    </tr>
    <tr>
      <th scope="row" colspan="2">Target class reached at atep 73</th>
      <th scope="row" colspan="2">Prediction: Yield</th>
    </tr>
    <tr>
      <th scope="row" colspan="2">Loss: -63.81779098510742</th>
      <th scope="row" colspan="2">Target Class Probability: 0.5005481243133545</th>
    </tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr>
      <th scope="col" colspan="4">Experiment 3</th>
    </tr>
    <tr>
      <th scope="col" colspan="2">Oryginal image</th>
      <th scope="col" colspan="2">Target image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">name: Yield</th>
      <th scope="row">label: 13</th>
      <th scope="row">name: Pedestrians</th>
      <th scope="row">label: 27</th>
    </tr>
    <tr>
      <th scope="row" colspan="4"><img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/ec8437e8-9b82-4dfe-95ee-864dcf78b88c" />
    </tr>
    <tr>
      <th scope="row" colspan="2">Target class reached at atep 57</th>
      <th scope="row" colspan="2">Prediction: Yield</th>
    </tr>
    <tr>
      <th scope="row" colspan="2">Loss: -48.407772064208984</th>
      <th scope="row" colspan="2">Target Class Probability: 0.5284264087677002</th>
    </tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr>
      <th scope="col" colspan="4">Experiment 4</th>
    </tr>
    <tr>
      <th scope="col" colspan="2">Oryginal image</th>
      <th scope="col" colspan="2">Target image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">name: Yield</th>
      <th scope="row">label: 13</th>
      <th scope="row">name: Stop</th>
      <th scope="row">label: 14</th>
    </tr>
    <tr>
      <th scope="row" colspan="4"><img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/2dc81c54-217a-4ba0-b85c-642c5c6f3a53" />
    </tr>
    <tr>
      <th scope="row" colspan="2">Target class reached at atep 425</th>
      <th scope="row" colspan="2">Prediction: Stop</th>
    </tr>
    <tr>
      <th scope="row" colspan="2">Loss: -407.3600158691406</th>
      <th scope="row" colspan="2">Target Class Probability: 0.5093487501144409</th>
    </tr>
  </tbody>
</table>


