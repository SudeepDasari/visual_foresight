template = """
<!DOCTYPE html>
<html>
<head>
<style>
table, th, td {{
  border: 1px solid black;
  border-collapse: collapse;
}}
th, td {{
  padding: 5px;
}}

td {{
  text-align: middle;
}}
</style>
</head>
<body>

<h2>{0}</h2>
<p>CEM Visualization Iter={1}, t={2}</p>

<table>
{3}
</table>

{4}
</body>
</html>
"""


def _format_title_row(title, length):
    template = "  <tr>\n    <th></th>\n"
    for i in range(length):
        template += "    <th> {}_{} </th>\n".format(title, i)
    template += "  </tr>\n"
    return template


def _format_img_row(name, path_list, height=128):
    template = "  <tr>\n    <td> <b> {} </b> </td>\n".format(name)
    for path in path_list:
        template += "    <td> <img src=\"{}\" height=\"{}\"> </td>\n".format(path, height)
    template += "  </tr>\n"
    return template


def _format_txt_row(name, content_list):
    template = "  <tr>\n    <td> <b> {} </b> </td>\n".format(name)
    for c in content_list:
        template += "    <td> {} </td>\n".format(c)
    template += "  </tr>\n"
    return template


def fill_template(cem_itr, t, item_dict, column_title='traj', exp_name='Visual MPC', img_height=128, extra_html=""):
    row_length = None
    for k, i in item_dict.items():
        if row_length is None:
            row_length = len(i)
        elif row_length != len(i):
            raise ValueError("All lengths should be the same")

    rows = ""
    rows += _format_title_row(column_title, row_length)
    for k, i in item_dict.items():
        if any([isinstance(i[0], str) and x in i[0] for x in ['gif', 'png', 'jpg']]):
            rows += _format_img_row(k, i, img_height)
        else:
            rows += _format_txt_row(k, i)

    return template.format(exp_name, cem_itr, t, rows, extra_html)


def save_gifs(save_worker, folder, name, gif_array):
    html_paths = []
    for i, a in enumerate(gif_array):
        html_path = 'assets/{}_{}.gif'.format(name, i)
        html_paths.append(html_path)
        save_worker.put(('mov', '{}/{}'.format(folder, html_path), a))
    return html_paths


def save_img(save_worker, folder, name, img):
    html_path = 'assets/{}.jpg'.format(name)
    save_worker.put(('img', '{}/{}'.format(folder, html_path), img))
    return html_path


def save_html(save_worker, path, content):
    save_worker.put(('txt_file', path, content))


def img_entry_html(img_path, height=640, caption=""):
    return "<br>\n<br>\n<div>\n  <img src=\"{0}\" height=\"{1}\">\n  <p>{2}</p>\n</div>".format(img_path,
                                                                                                height, caption)
