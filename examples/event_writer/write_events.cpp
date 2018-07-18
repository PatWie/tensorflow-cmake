#include <tensorflow/core/util/events_writer.h>
#include <string>
#include <iostream>


void write_scalar(tensorflow::EventsWriter* writer, double wall_time, tensorflow::int64 step,
                  const std::string& tag, float simple_value) {
  tensorflow::Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_simple_value(simple_value);
  writer->WriteEvent(event);
}


int main(int argc, char const *argv[]) {

  std::string envent_file = "./events";
  tensorflow::EventsWriter writer(envent_file);
  for (int i = 0; i < 150; ++i)
    write_scalar(&writer, i * 20, i, "loss", 150.f / i);

  return 0;
}