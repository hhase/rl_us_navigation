#include "tracking_plug_in/controller.hpp"        // for PluginController
#include <ImFusion/GUI/AlgorithmController.h>  // for AlgorithmController
#include <qobjectdefs.h>                       // for SIGNAL, SLOT
#include <qpushbutton.h>                       // for QPushButton
#include <memory>                              // for make_shared, shared_ptr
#include "tracking_plug_in/algorithm.hpp"         // for PluginAlgorithm
#include "ui_controller.h"                        // for Ui_Demo

#include <ImFusion/Base/DataList.h>

namespace ImFusion {

namespace TrackingPlugIn {
PluginController::PluginController(PluginAlgorithm* algorithm)
        : AlgorithmController(algorithm), algorithm_(algorithm) {
    ui_ = std::make_shared<Ui_Controller>();
    ui_->setupUi(this);
}

void PluginController::init() {
    // ImFusion standard command.
    addToAlgorithmDock();

    //***GUI SET UP***
 // connect(m_button, SIGNAL (released()),this, SLOT (handleButton()));
    connect(ui_->dummy_button, &QPushButton::clicked, this, &PluginController::onButtonClicked);
}

void PluginController::onButtonClicked() {

    ImFusion::DataList input;
    input = m_main->selectedData();
    std::string path = ui_->textEdit_path->toPlainText().toUtf8().constData();
    UltrasoundSweep* sweep = static_cast<UltrasoundSweep*>(input[0]);
    algorithm_->exportSweep(sweep, path);

    //algorithm_->exportSweep(sweep, path);
}


}  // namespace TrackingPlugIn.
}  // namespace ImFusion.
