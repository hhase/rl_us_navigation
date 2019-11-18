#pragma once

#include <ImFusion/Base/Algorithm.h>  // for IoAlgorithm
#include <ImFusion/Base/DataList.h>   // for DataList
#include <ImFusion/Base/TrackingStream.h>
#include <ImFusion/IO/PngIO.h>
#include <ImFusion/US/UltrasoundSweep.h>
#include <iostream>

namespace ImFusion {
class Algorithm;
class Properties;

namespace TrackingPlugIn {

class PluginAlgorithm : public ImFusion::Algorithm {
 public:
  /// Creates the algorithm instance
  PluginAlgorithm();

  static bool createCompatible(const DataList& data, Algorithm** a = nullptr);

  /// Applies the processing
  void compute() override;

  /// Make the resulting data available here
  void output(DataList& data_out) override;

  /// Methods implementing the Configurable interface
  void configure(const Properties* p) override;
  void configuration(Properties* p) const override;

  void changeMember();

  void exportSweep(UltrasoundSweep* sweep, std::string path);

 private:
  int placeholder_{0};
};

}  // namespace TrackingPlugIn.
}  // namespace ImFusion.
