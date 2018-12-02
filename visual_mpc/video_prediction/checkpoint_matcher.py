import tensorflow as tf


def variable_checkpoint_matcher(conf, vars, model_file=None, ignore_varname_firstag=False):
  """
  for every variable in vars takes its name and looks inside the
  checkpoint to find variable that matches its name beginning from the end
  :param vars:
  :return:
  """
  if model_file is None:
    ckpt = tf.train.get_checkpoint_state(conf['output_dir'])
    model_file = ckpt.model_checkpoint_path

  print('variable checkpoint matcher using model_file:',model_file)

  reader = tf.train.NewCheckpointReader(model_file)
  var_to_shape_map = reader.get_variable_to_shape_map()
  check_names = list(var_to_shape_map.keys())

  vars = dict([(var.name.split(':')[0], var) for var in vars])
  new_vars = {}
  for varname in list(vars.keys()):
    found = False
    for ck_name in check_names:
      ck_name_parts = ck_name.split('/')
      varname_parts = varname.split('/')

      if ignore_varname_firstag:
        varname_parts = varname_parts[1:]

      if varname_parts == ck_name_parts[-len(varname_parts):]:
        new_vars[ck_name] = vars[varname]
        found = True
        # print("found {} in {}".format(varname, ck_name))
        break
    if not found:
      raise ValueError("did not find variable {}".format(varname))
  return new_vars