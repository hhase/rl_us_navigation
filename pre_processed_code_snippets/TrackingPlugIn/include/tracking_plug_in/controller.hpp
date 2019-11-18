#pragma once

#include <ImFusion/GUI/AlgorithmController.h>  // for AlgorithmController
#include <ImFusion/US/UltrasoundSweep.h>
#include <qobjectdefs.h>  // for Q_OBJECT, slots
#include <qstring.h>      // for QString
#include <qwidget.h>      // for QWidget
#include <memory>         // for shared_ptr

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/GUI/AnnotationModel.h>
#include <ImFusion/GUI/MainWindowBase.h>

class QObject;
class Ui_Controller;  // lines 10-10

namespace ImFusion {
namespace TrackingPlugIn {

class PluginAlgorithm;

class PluginController : public QWidget, public AlgorithmController {
  Q_OBJECT
 public:
  PluginController(PluginAlgorithm* algorithm);
  virtual ~PluginController() override = default;

  // Initializes the widget
  void init() override;

 public slots:
  void onButtonClicked();

 protected:
  std::shared_ptr<Ui_Controller> ui_{nullptr};  ///< The actual GUI
  PluginAlgorithm* algorithm_{nullptr};         ///< The algorithm instance
};

}  // namespace TrackingPlugIn.
}  // namespace ImFusion.
