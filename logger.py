import os

class logger:

    def __init__(self, root_log_path):
        self.root_log_path = root_log_path
        self.cur_log_path = root_log_path
        self.spliter = ', '
        self.k_v_pointer = ' : '

    def create_path(self, log_path):
        if not os.path.exists(self.root_log_path):
            os.makedirs(self.root_log_path)

        cur_log_path = os.path.join(self.root_log_path, log_path)

        if not os.path.exists(cur_log_path):
            os.makedirs(cur_log_path)

        self.cur_log_path = cur_log_path

    def open_log_file(self, file_name):
        f = open(os.path.join(self.cur_log_path, file_name), 'a')
        return f

    def close_log_file(self, f):
        f.close()

    def log_line(self, file_name, contents, step_line=False):
        f = open(os.path.join(self.cur_log_path, file_name), 'a')
        f.write(contents)
        if step_line:
            f.write('\n')
        f.close()


if __name__ == '__main__':
    logger = logger('save_model/vTB')
    logger.create_path('PURE DDPG')
    f = logger.open_log_file('exp1.txt')
    contents = 'asdc fefcaf'
    logger.log_line(f, contents)
    logger.close_log_file(f)
