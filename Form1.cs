using System;
using System.Windows.Forms;

namespace RadioButtonApp
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            InitializeCustomComponents();
        }

        private void InitializeCustomComponents()
        {
           
            labelTitle.Text = "GIS程序设计";
            labelTitle.Font = new System.Drawing.Font("Microsoft YaHei", 16F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            labelTitle.AutoSize = false;
            labelTitle.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;

            groupBoxGrade.Visible = false;
            groupBoxClass.Visible = false;

            comboGroup.Text = "（组号）";

            
            radioGrade.CheckedChanged += new EventHandler(RadioButton_CheckedChanged);
            radioClass.CheckedChanged += new EventHandler(RadioButton_CheckedChanged);

            
            comboGroup.SelectedIndexChanged += new EventHandler(ComboGroup_SelectedIndexChanged);
        }

        private void RadioButton_CheckedChanged(object sender, EventArgs e)
        {
            if (radioGrade.Checked)
            {
                groupBoxGrade.Visible = true;
                groupBoxClass.Visible = false;
            }
            else if (radioClass.Checked)
            {
                groupBoxClass.Visible = true;
                groupBoxGrade.Visible = false;
            }
        }

        private void ComboGroup_SelectedIndexChanged(object sender, EventArgs e)
        {
            
            string selectedGroup = comboGroup.SelectedItem.ToString();
            MessageBox.Show($"您选择的是: {selectedGroup}", "组号选择", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
