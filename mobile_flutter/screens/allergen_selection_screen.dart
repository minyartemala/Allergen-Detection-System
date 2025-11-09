// screens/allergen_selection_screen.dart
import 'package:flutter/material.dart';
import 'scanner_screen.dart';

class AllergenSelectionScreen extends StatefulWidget {
  const AllergenSelectionScreen({Key? key}) : super(key: key);

  @override
  _AllergenSelectionScreenState createState() => _AllergenSelectionScreenState();
}

class _AllergenSelectionScreenState extends State<AllergenSelectionScreen> {
  List<String> selectedAllergens = [];

  final List<Map<String, String>> allergens = [
    {'fr': 'Gluten', 'en': 'gluten', 'ar': 'الغلوتين'},
    {'fr': 'Crustacés', 'en': 'crustaceans', 'ar': 'القشريات'},
    {'fr': 'Œufs', 'en': 'eggs', 'ar': 'بيض'},
    {'fr': 'Poissons', 'en': 'fish', 'ar': 'الأسماك'},
    {'fr': 'Arachides', 'en': 'peanuts', 'ar': 'الفول السوداني'},
    {'fr': 'Fruits à coque', 'en': 'tree nuts', 'ar': 'المكسرات'},
    {'fr': 'Graines de sésame', 'en': 'sesame seeds', 'ar': 'بذور السمسم'},
    {'fr': 'Lait', 'en': 'milk', 'ar': 'الحليب'},
    {'fr': 'Moutarde', 'en': 'mustard', 'ar': 'الخردل'},
    {'fr': 'Céleri', 'en': 'celery', 'ar': 'الكرفس'},
    {'fr': 'Légumineuses', 'en': 'legumes', 'ar': 'البقوليات'},
    {'fr': 'Sulfites', 'en': 'sulphites', 'ar': 'الكبريتات'},
    {'fr': 'Lupin', 'en': 'lupin', 'ar': 'الترمس'},
    {'fr': 'Mollusques', 'en': 'mollusks', 'ar': 'الرخويات'},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F5DC),
      appBar: AppBar(
        title: const Text(
          'Sélectionnez vos allergènes',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        backgroundColor: const Color(0xFF8FBC8F),
        iconTheme: const IconThemeData(color: Colors.white),
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            const SizedBox(height: 25),
            const Text(
              'Choisissez les allergènes que vous souhaitez détecter :',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: Color(0xFF8FBC8F),
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 15),

            if (selectedAllergens.isNotEmpty)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: const Color(0xFF8FBC8F).withOpacity(0.1),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFF8FBC8F)),
                ),
                child: Text(
                  '${selectedAllergens.length} allergène(s) sélectionné(s)',
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF8FBC8F),
                  ),
                ),
              ),
            const SizedBox(height: 15),
            Expanded(
              child: ListView.builder(
                itemCount: allergens.length,
                itemBuilder: (context, index) {
                  final allergen = allergens[index];
                  final isSelected = selectedAllergens.contains(allergen['fr']);

                  return Container(
                    margin: const EdgeInsets.symmetric(vertical: 6),
                    child: Card(
                      elevation: isSelected ? 10 : 3,
                      color: isSelected ? const Color(0xFF8FBC8F) : Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15),
                        side: BorderSide(
                          color: isSelected ? const Color(0xFF8FBC8F) : Colors.grey.shade300,
                          width: 4,
                        ),
                      ),
                      child: ListTile(
                        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                        title: Text(
                          allergen['fr']!,
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 16,
                            color: isSelected ? Colors.white : const Color(0xFF8FBC8F),
                          ),
                        ),
                        subtitle: Text(
                          allergen['ar']!,
                          style: TextStyle(
                            color: isSelected ? Colors.white70 : Colors.grey.shade600,
                            fontSize: 14,
                          ),
                        ),
                        trailing: isSelected
                            ? const Icon(Icons.check_box, color: Colors.white, size: 25)
                            : const Icon(Icons.check_box_outline_blank, color: Color(0xFF8FBC8F)),
                        onTap: () {
                          setState(() {
                            if (isSelected) {
                              selectedAllergens.remove(allergen['fr']);
                            } else {
                              selectedAllergens.add(allergen['fr']!);
                            }
                          });
                        },
                      ),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 20),

            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () {
                      setState(() {
                        if (selectedAllergens.length == allergens.length) {
                          selectedAllergens.clear();
                        } else {
                          selectedAllergens = allergens.map((a) => a['fr']!).toList();
                        }
                      });
                    },
                    style: OutlinedButton.styleFrom(
                      foregroundColor: const Color(0xFF8FBC8F),
                      side: const BorderSide(color: Color(0xFF8FBC8F), width: 2),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(25),
                      ),
                    ),
                    child: Text(
                      selectedAllergens.length == allergens.length
                          ? 'Tout désélectionner'
                          : 'Tout sélectionner',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 15),
            SizedBox(
              width: double.infinity,
              height: 60,
              child: ElevatedButton(
                onPressed: selectedAllergens.isNotEmpty
                    ? () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => ScannerScreen(selectedAllergens: selectedAllergens),
                    ),
                  );
                }
                    : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF8FBC8F),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                  elevation: 5,
                ),
                child: const Text(
                  'Continuer vers le scanner',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}