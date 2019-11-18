#include "tracking_plug_in/factory.hpp"
#include <ImFusion/Base/Algorithm.h>     // for Algorithm
#include <vector>                        // for vector
#include "tracking_plug_in/algorithm.hpp"   // for PluginAlgorithm
#include "tracking_plug_in/controller.hpp"  // for PluginController

namespace ImFusion {
class AlgorithmController;

namespace TrackingPlugIn {

PluginAlgorithmFactory::PluginAlgorithmFactory(){
    registerAlgorithm<PluginAlgorithm>("MyCategory;Tracking Plugin");
}


AlgorithmController* PluginControllerFactory::create(
    ImFusion::Algorithm* a) const {
  if (PluginAlgorithm* algorithm = dynamic_cast<PluginAlgorithm*>(a)) {
    return new PluginController(algorithm);
  }
  return nullptr;
}

}  // namespace TrackingPlugIn.
}  // namespace ImFusion.
