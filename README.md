# Resceleb

## How to use

In config.yaml there is a list of attributes. These attributes are ordered in the same order of the columns in celebA. They must remain in that order. 
Also in config are two other lists. These control the attributes used as targets for the 'hair_color' and 'hair_style' options. If a sample is negative for all targets 
they are assigned to an 'other' valued target.

Also in config
  - log directory to use
  - path to celebA dataset


Supports distributed training over multiple GPUs.


#### Arguments
- mode: either 'hair_style' or 'hair_color'. Changes which targets are used.
- device: a device of list of devices to use (default: cpu)
- optimizer: Adam or SGD (Default Adam)
- lr
- epochs
- batch size
- seed

