from cppbriann.simple.build import simple_module as m

class BLub():
    """Test comment"""

    def __init__(self):
        self.name = "BLub"

a = 10
b = 20

result = m.simple_cpp_function(a, b)

print(f"a = {a}, b = {b}, result = {result}")


from cppbriann.advanced.build import advanced_module as m

department = m.Department.HR

print(f"We are in the {department} department")

employee = m.Person("John", "Manager", m.Department.HR)

print(f"{employee.name} is a {employee.position} in the {employee.department} department")
print("Reassigning employee to the Engineering department")
employee.ReassignDepartment(m.Department.Engineering)

print(f"{employee.name} is a {employee.position} in the {employee.department} department")