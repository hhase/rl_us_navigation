/********************************************************************************
** Form generated from reading UI file 'controller.ui'
**
** Created by: Qt User Interface Compiler version 5.9.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONTROLLER_H
#define UI_CONTROLLER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Controller
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *main_box;
    QVBoxLayout *verticalLayout_6;
    QPushButton *dummy_button;
    QTextEdit *textEdit_path;

    void setupUi(QWidget *Controller)
    {
        if (Controller->objectName().isEmpty())
            Controller->setObjectName(QStringLiteral("Controller"));
        Controller->resize(263, 102);
        verticalLayout = new QVBoxLayout(Controller);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        main_box = new QGroupBox(Controller);
        main_box->setObjectName(QStringLiteral("main_box"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(main_box->sizePolicy().hasHeightForWidth());
        main_box->setSizePolicy(sizePolicy);
        verticalLayout_6 = new QVBoxLayout(main_box);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        dummy_button = new QPushButton(main_box);
        dummy_button->setObjectName(QStringLiteral("dummy_button"));

        verticalLayout_6->addWidget(dummy_button);

        textEdit_path = new QTextEdit(main_box);
        textEdit_path->setObjectName(QStringLiteral("textEdit_path"));

        verticalLayout_6->addWidget(textEdit_path);


        verticalLayout->addWidget(main_box);


        retranslateUi(Controller);

        QMetaObject::connectSlotsByName(Controller);
    } // setupUi

    void retranslateUi(QWidget *Controller)
    {
        Controller->setWindowTitle(QApplication::translate("Controller", "Form", Q_NULLPTR));
        main_box->setTitle(QApplication::translate("Controller", "Demo Plugin", Q_NULLPTR));
        dummy_button->setText(QApplication::translate("Controller", "Extract Sweep Data", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class Controller: public Ui_Controller {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONTROLLER_H
