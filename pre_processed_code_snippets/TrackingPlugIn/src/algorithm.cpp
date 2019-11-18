#include "tracking_plug_in/algorithm.hpp"
#include <ImFusion/Base/AlgorithmListener.h>  // for AlgorithmListener
#include <ImFusion/Base/Log.h>                // for LOG_INFO
#include <ImFusion/Base/Properties.h>         // for Properties
#include <vector>                             // for vector
#include <fstream>
#include <QtWidgets/QWidget>

namespace ImFusion {
class Algorithm;

namespace TrackingPlugIn {

PluginAlgorithm::PluginAlgorithm() { configureDefaults(); }

bool PluginAlgorithm::createCompatible(const DataList& data, Algorithm** a) {
  // Check requirements to create the algorithm
  if (data.size() != 0) {
    return false;
  }

  // Requirements are met, create the algorithm if asked
  if (a) {
    *a = new PluginAlgorithm();
  }
  return true;
}

void PluginAlgorithm::compute() {}

void PluginAlgorithm::output(DataList& /*unused*/) {}

// This methid restors the member parameters when a workspace file is loaded.
// We can store previous configurations in a workspace file and load them here.
void PluginAlgorithm::configure(const Properties* p) {
  if (p == nullptr) {
    return;
  }
  p->param("placeholder", placeholder_);
  // Notify all listeners that  parameters have changed.
  for (auto& listener : m_listeners) {
    listener->algorithmParametersChanged();
  }
}

// Retrieve member variable values from saved workspace.
void PluginAlgorithm::configuration(Properties* p) const {
  // This method is necessary to store our settings in a workspace file.
  if (p == nullptr) {
    return;
  }
  p->setParam("placeholder", placeholder_, 0);
}

void PluginAlgorithm::exportSweep(UltrasoundSweep* sweep, std::string path) {
    // Open ImFusion stream
    TrackingStream* _trackStream = sweep->tracking(0);

    for (auto imageIndex = 0; imageIndex < sweep->size(); imageIndex++) {
        double timestamp = sweep->timestamp(imageIndex);
        SharedImage* image = sweep->get(imageIndex);
        mat4 _frameMat = _trackStream->matrix(timestamp);
        std::cout << "Frame " << imageIndex << std::endl
                  << "Mat" << std::endl
                  << _frameMat << std::endl;

        //PngIO::writeImage(path + "_" + std::string(imageIndex), image, 3);

        PngIO::writeImage(path + "frames/_" + std::to_string(imageIndex), image->mem());
        std::ofstream myfile;
        myfile.open(path + "_mat.txt", std::ios::app);
        myfile << path << "Frame_" + std::to_string(imageIndex)+"\n";
        myfile << path << _frameMat << "\n";
        myfile.close();
    }
}


void PluginAlgorithm::changeMember() {
  ++placeholder_;
  LOG_INFO("changeMember() called!");
}

}  // namespace TrackingPlugIn.
}  // namespace ImFusion
