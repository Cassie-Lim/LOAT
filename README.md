# Advancing Object Goal Navigation Through LLM-enhanced Object Affinities Transfer

[Advancing Object Goal Navigation Through LLM-enhanced Object Affinities Transfer](https://arxiv.org/abs/2403.09971)<br />

## Setting up the environment

### Basics

1. Clone the repo

   ```bash
   git clone ~
   ```

2. Create the conda environmnet

   ```bash
   conda env create --file=environments.yml
   conda activate loat
   # install habitat
   git clone https://github.com/facebookresearch/habitat-lab.git
   cd habitat-lab; git checkout tags/v0.1.5; 
   pip install -e .
   ```

3. Additional preliminaries to use ALFRED scenes

   - You will need to download alfred scene data and process it using the [alfred repo]().

   - The last step asks to create a soft link from `json_2.1.0` to `alfred_data_all`. However, in our experience we had to link from `alfred_feat_2.1.0`. The target folder is still the same, though (`json_2.1.0`).

     ```bash
     # before
     ln -s $ALFRED_ROOT/data/json_2.1.0 $FILM/alfred_data_all
     
     # after
     ln -s $ALFRED_ROOT/data/json_feat_2.1.0/ $FILM/alfred_data_all/json_2.1.0
     ```
     
   4. After this step, `alfred_data_all` directory should look like this:

      ```bash
      alfred_data_all
         └── json_2.1.0
         ├── tests_unseen
            ├── tests_seen
            ├── valid_unseen
            ├── tests_seen
            ├── trial_T2019...
            └── ...
      ```

4. Download Trained models

   1. Download "Pretrained_Models_FILM" from [this link](https://drive.google.com/file/d/1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa/view?usp=sharing) kindly provided by FILM's author

   2. Relocate the downloaded models to the correct directories

      ```
      mv Pretrained_Models_FILM/maskrcnn_alfworld models/segmentation/maskrcnn_alfworld
      mv Pretrained_Models_FILM/depth_models models/depth/depth_models
      mv Pretrained_Models_FILM/new_best_model.pt models/semantic_policy/best_model_multi.pt
      ```
   3. Download the model weight for pretrained LOAT-P semantic policy from [here](https://drive.google.com/drive/folders/1XSB77pWJmC8NA8INVbCofP918fdoikEk?usp=drive_link). Move the downloaded weights to `models/semantic_policy/cap_dict/`
### Running with headless machines

Running ALFRED environment on headless machines is not straightforward, and it heavily depends on your hardware configuration. The root of the problem is that ALFRED is based on Unity, which requires some sort of display to operate properly. Unfortunately, headless machines do not have displays, so it must be substituted by a virtual display, which is not straightforward to set up.

We will introduce here what worked for us; please also refer to discussions in [FILM's repo](https://github.com/soyeonm/FILM), [ALFRED's repo](https://github.com/askforalfred/alfred), and [ALFRED worlds's repo](https://github.com/alfworld/alfworld) if it does not work for you.

1. You need to first run a Xserver with 

   ```
   sudo python alfred_utils/scripts/startx.py 0
   ```

   - The number after `startx.py` indicates the ID in which you will set up the virtual display. If something is already running on that ID, you will receive an error `Cannot establish any listening sockets - Make sure an X server isn't already running(EE)`. Try other numbers until you find the one that works.

   - If you set a Xdisplay other than 0 (if you ran `python alfred_utils/scripts/startx.py 1`, for example), set the environmental variable `DISPLAY` accordingly.

     ```
     export DISPLAY=:1
     ```

   - If you get an error: `Only console users are allowed to run the X server`, add the following line in `/etc/X11/Xwrapper.config`

     ```
     allowed_users = anybody
     ```

2. Check that the display is accessible

   ```
   glxinfo | grep rendering
   ```

3. Start another terminal session and run the following to evaluate the model. `--x_display` value must match the display ID you set in step 1. 

   ```bash
   python main.py -n1 --max_episode_length 1000 --num_local_steps 25 --num_processes 1 --eval_split valid_unseen --from_idx 0 --to_idx 510 --max_fails 10 --debug_local --learned_depth --use_sem_seg --set_dn testrun -v 0 --which_gpu 0 --x_display 0 --sem_policy_type mlm --mlm_fname mlmscore_equal --mlm_options aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay --seed 1 --splits alfred_data_small/splits/oct21.json --grid_sz 240 --mlm_temperature 1 --approx_last_action_success --language_granularity high --centering_strategy local_adjustment --target_offset_interaction 0.5 --obstacle_selem 9 --debug_env
   ```

   - Do not enable `-v 1` on a headless machine.
   - This will run the evaluation on debug mode, which provides more readable error messages.
   - Details on the arguments for `main.py` is summarized [here](evaluation.md).



## Training / Evaluation
> TODO: this part haven't be tidied yet
To train the semantic policy of LOAT-P from scratch, you could start with downloading the dataset used in FILM from [this link](https://drive.google.com/file/d/1TWxKSAxvYKA8hi1RyUgQ0CGmLSe4UR_n/view?usp=sharing) and extract it as `models/semantic_policy/data/maps`.

Then run `python models/semantic_policy/preprocess_data.py` to carry out the necessary preprocess.

Run the following command for training:
```
python models/semantic_policy/my_train_map_multi.py --eval_freq 50 --dn YOUR_DESIRED_NAME --seed YOUR_SEED --lr 0.001 --num_epochs 1 
```

Evaluation and leaderboard submission is summarized in [this document](evaluation.md).



## Helpful links

- [List of Arguments for main.py](evaluation.md)
- [Debugging tips](debugging.md)
- [Obtaining MLM scores]() → coming soon




## Acknowledgements

Our codebase is heavily based on [So Yeon Min's FILM ](https://github.com/soyeonm/FILM) and [Yuki Inoue's Prompter respository](https://github.com/hitachi-rd-cv/prompter-alfred).

FILM repository borrows ideas from the following repositories:

- Most of [alfred_utils](https://github.com/soyeonm/FILM/control_helper/alfred_utils) comes from [Mohit Shridhar's ALFRED repository](https://github.com/askforalfred/alfred).
- [models/depth](https://github.com/soyeonm/FILM/tree/public/models/depth) comes from [Valts Blukis' HLSM](https://github.com/valtsblukis/hlsm).
- Code for semantic mapping comes from [Devendra Singh Chaplot's OGN](https://github.com/devendrachaplot/OGN)



If you intend to utilize this repository or derive inspiration from its contents, we kindly request that you cite our paper:

```
@misc{lin2024advancing,
      title={Advancing Object Goal Navigation Through LLM-enhanced Object Affinities Transfer}, 
      author={Mengying Lin and Yaran Chen and Dongbin Zhao and Zhaoran Wang},
      year={2024},
      eprint={2403.09971},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```